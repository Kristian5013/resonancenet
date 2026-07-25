"""The seam between the consensus tensor layout and one PyTorch module tree.

The layout in `consensus/init.py` is the authority: it fixes which parameter is
which, and the payload a worker submits is the concatenation of tensors in that
order. PyTorch names its parameters after whatever the module attributes happen to
be called, and `sorted(model.named_parameters())` produces an entirely different
permutation of the same tensors — on regtest, zero of 46 positions agree, and with
sixteen layers it also orders them 0, 1, 10, 11, …, 2 because the sort is textual.

That mismatch has no symptom. Byte *i* of a payload would mean one parameter to
the worker and a different one to every C++ node; aggregation would still run,
hashes would still be produced, and the model would train into noise. Nothing on
the wire can detect it — only this file and the cross-language test can.

So the mapping is explicit and total. Every consensus tensor must resolve to
exactly one PyTorch parameter of exactly the right shape, and every PyTorch
parameter must be claimed by exactly one consensus tensor. Anything unmatched
raises rather than being skipped: a silently dropped tensor is the same failure
wearing a different mask.
"""

from __future__ import annotations

import hashlib

import numpy as np
import torch

from .consensus import init as consensus_init


class LayoutError(RuntimeError):
    """The module tree and the consensus layout disagree. Never recoverable."""


def torch_name_for(consensus_name: str) -> str:
    """The PyTorch parameter name holding a given consensus tensor.

    Written as an explicit translation rather than a naming convention, because a
    convention silently absorbs a renamed module and this must not.
    """
    if consensus_name == "tok_embeddings":
        return "tok.weight"
    if consensus_name == "norm":
        return "norm.weight"
    if consensus_name == "output":
        return "head.weight"

    if not consensus_name.startswith("layers."):
        raise LayoutError(f"layout: no PyTorch parameter is defined for '{consensus_name}'")
    _, index, rest = consensus_name.split(".", 2)

    suffixes = {
        "attention_norm": "ln1.weight",
        "attention.wq": "attn.wq.weight",
        "attention.wk": "attn.wk.weight",
        "attention.wv": "attn.wv.weight",
        "attention.wo": "attn.wo.weight",
        "attention.q_norm": "attn.q_norm.weight",
        "attention.k_norm": "attn.k_norm.weight",
        "ffn_norm": "ln2.weight",
        "feed_forward.w_gate": "mlp.w_gate.weight",
        "feed_forward.w_up": "mlp.w_up.weight",
        "feed_forward.w_down": "mlp.w_down.weight",
    }
    if rest not in suffixes:
        raise LayoutError(f"layout: no PyTorch parameter is defined for '{consensus_name}'")
    return f"blocks.{index}.{suffixes[rest]}"


def consensus_parameters(model, model_spec) -> list[tuple[str, torch.nn.Parameter]]:
    """The model's parameters in consensus layout order.

    Replaces the alphabetical ordering entirely. Never call
    `sorted(model.named_parameters())` on a path that produces or consumes a
    payload.
    """
    layout = consensus_init.tensor_layout(model_spec)
    by_name = dict(model.named_parameters())

    ordered: list[tuple[str, torch.nn.Parameter]] = []
    claimed: set[int] = set()
    for tensor in layout:
        torch_name = torch_name_for(tensor.name)
        if torch_name not in by_name:
            raise LayoutError(
                f"layout: consensus tensor '{tensor.name}' expects PyTorch parameter "
                f"'{torch_name}', which this model does not have"
            )
        parameter = by_name[torch_name]
        if parameter.numel() != tensor.elements:
            raise LayoutError(
                f"layout: '{tensor.name}' should hold {tensor.elements} elements "
                f"({tensor.rows}x{tensor.cols}), but '{torch_name}' holds {parameter.numel()}"
            )
        ordered.append((tensor.name, parameter))
        claimed.add(id(parameter))

    # Every parameter must be claimed. A tensor the layout does not know about
    # would be trained and then silently dropped from every contribution — the
    # model would drift away from what the network is aggregating, one round at a
    # time.
    #
    # Tied weights legitimately appear under two names and one identity, so the
    # comparison is by identity, not by name.
    unclaimed = sorted(name for name, p in by_name.items() if id(p) not in claimed)
    if unclaimed:
        raise LayoutError(
            f"layout: {len(unclaimed)} PyTorch parameter(s) are outside the consensus layout: "
            f"{', '.join(unclaimed[:5])}"
        )

    total = sum(p.numel() for _, p in ordered)
    if total != model_spec.parameter_count():
        raise LayoutError(
            f"layout: ordered parameters hold {total} elements, the spec says "
            f"{model_spec.parameter_count()}"
        )
    return ordered


@torch.no_grad()
def load_genesis_weights(model, model_spec, round_id_hash: bytes) -> None:
    """Fills the model with the initial weights the network agreed on.

    Without this the worker trains from PyTorch's default initialisation, which is
    reproducible only within one PyTorch version and has nothing to do with the
    pinned genesis weights anchor. Every node must start from the same state or the
    deltas they compute cannot be aggregated or verified — and the failure looks
    like every worker cheating at once.
    """
    seed = consensus_init.init_seed(round_id_hash)
    layout = consensus_init.tensor_layout(model_spec)
    ordered = consensus_parameters(model, model_spec)

    for index, ((name, parameter), tensor) in enumerate(zip(ordered, layout)):
        if name != tensor.name:
            raise LayoutError(f"layout: ordering drifted at index {index}: {name} != {tensor.name}")
        values = consensus_init.init_tensor_slice(seed, index, tensor, 0, tensor.elements)
        flat = torch.from_numpy(np.ascontiguousarray(values))
        parameter.copy_(flat.view(parameter.shape).to(parameter.dtype))


@torch.no_grad()
def consensus_weights_hash(model, model_spec, slice_elements: int = 1 << 22) -> bytes:
    """SHA3-256 of the weights, in the form consensus defines them.

    Every tensor in layout order, each element as big-endian bf16, and nothing
    else — no names, no shapes, no separators. The hash a node computes over a
    checkpoint must be reachable by a worker holding the same weights, so this is
    the only weights hash that may be compared against `base_weights_hash` or the
    genesis anchor.
    """
    hasher = hashlib.sha3_256()
    for _, parameter in consensus_parameters(model, model_spec):
        flat = parameter.detach().reshape(-1)
        for start in range(0, flat.numel(), slice_elements):
            chunk = flat[start : start + slice_elements].to(torch.float32).cpu().numpy()
            hasher.update(consensus_init.float_to_bf16(chunk).astype(">u2").tobytes())
    return hasher.digest()


@torch.no_grad()
def consensus_weights_bytes(model, model_spec, slice_elements: int = 1 << 22) -> bytes:
    """The weights in the exact form consensus defines them.

    Every tensor in layout order, each element as big-endian bf16, nothing else.
    These are the bytes a node stores and serves, and the bytes whose hash the
    checkpoint commits to — so this and `consensus_weights_hash` must walk the
    model identically, which is why they share `consensus_parameters`.
    """
    out = bytearray()
    for _, parameter in consensus_parameters(model, model_spec):
        flat = parameter.detach().reshape(-1)
        for start in range(0, flat.numel(), slice_elements):
            chunk = flat[start : start + slice_elements].to(torch.float32).cpu().numpy()
            out += consensus_init.float_to_bf16(chunk).astype(">u2").tobytes()
    return bytes(out)


@torch.no_grad()
def load_consensus_weights(model, model_spec, raw: bytes) -> None:
    """Loads weights from their canonical byte form.

    How a worker joins a network that has already moved: it fetches the bytes the
    checkpoint names, verifies them against that hash, and loads them here. Without
    this, catching up would need every historical payload — which the retention
    window deliberately does not keep.
    """
    expected = model_spec.parameter_count() * 2
    if len(raw) != expected:
        raise LayoutError(f"layout: weights are {len(raw)} bytes, the model needs {expected}")

    # bf16 to float32 is exact: the low sixteen bits are zero, so nothing is lost
    # and nothing is rounded. A worker that loaded these and hashed them again must
    # reach the same value it verified.
    bits = np.frombuffer(raw, dtype=">u2").astype(np.uint32) << np.uint32(16)
    values = bits.view(np.uint32).astype(np.uint32).tobytes()
    flat_all = np.frombuffer(values, dtype=np.float32)

    cursor = 0
    for _, parameter in consensus_parameters(model, model_spec):
        n = parameter.numel()
        # Copied out of the read-only buffer: torch refuses to own a non-writable
        # array, and silently aliasing one would make a later in-place update
        # scribble on bytes that are supposed to be immutable.
        piece = np.array(flat_all[cursor : cursor + n], dtype=np.float32)
        parameter.copy_(torch.from_numpy(piece).view(parameter.shape).to(parameter.dtype))
        cursor += n
    if cursor * 2 != len(raw):
        raise LayoutError("layout: the model consumed a different number of weights than arrived")


@torch.no_grad()
def apply_consensus_update(model, model_spec, update, scale_exp: int) -> bytes:
    """Applies an aggregated outer update and returns the new consensus weights hash.

    This is the worker's half of producing a checkpoint. The node aggregates — it
    holds the payloads — but it cannot apply the result, because the weights live
    here in GPU memory and copying a multi-gigabyte model into the daemon to
    compute one hash would put it in the wrong place.

    What happens here is a COMPUTATION, not a judgement: anyone holding the same
    base weights and the same update reaches the same hash. That is precisely what
    makes the resulting checkpoint checkable by every other node, and it is why the
    worker is allowed to perform it despite not being trusted to decide anything.

    Sign convention matches `inner_loop.apply_update`: the pseudo-gradient is
    (before - after), so it already points downhill and the update is SUBTRACTED.
    """
    inv = 2.0 ** (-scale_exp)
    update = np.asarray(update, dtype=np.int64)

    cursor = 0
    for _, parameter in consensus_parameters(model, model_spec):
        n = parameter.numel()
        chunk = torch.from_numpy(update[cursor : cursor + n].astype(np.float64))
        delta = (chunk * inv).to(torch.float32).reshape(parameter.shape)
        parameter.sub_(delta.to(parameter.device, parameter.dtype))
        cursor += n
    if cursor != len(update):
        # A length mismatch means the node and this worker disagree about the
        # model, which would corrupt every parameter after the point they diverge.
        raise LayoutError(
            f"layout: the update holds {len(update)} values, the model has {cursor} parameters"
        )
    return consensus_weights_hash(model, model_spec)
