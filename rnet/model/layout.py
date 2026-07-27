"""The canonical order of a model's tensors — names, shapes, and which shard
holds them.

Everything downstream depends on this order existing and being total. The
weights hash is a hash of the tensors laid end to end, so two nodes that
disagree about the order disagree about the hash while holding identical
weights. The initial weights are derived from a keyed stream indexed by
position in this layout, so the order does not merely describe the model — it
partly defines it.

TORCH-FREE ON PURPOSE. This module and everything it feeds must run on a
machine with no deep-learning stack, because deciding whether a network's
weights are what it claims should not require the hardware to train them. The
tensors themselves are built in `transformer.py`, which does import torch.

THE ORDER IS: embeddings, then layers in index order, then the final norm, then
the untied head if there is one. Inside a layer the order is attention norm,
q, k, v, o, the qk norms if present, the feed-forward norm, then the
feed-forward — which is either one dense triple or a router followed by every
expert, routed first and shared after.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import prod

from ..consensus.model_spec import ModelSpec


class Kind(Enum):
    """How a tensor is initialised, which is the only thing the layout needs to
    know about it.

    NORM tensors start at one and everything else starts from the keyed stream
    scaled by its fan-in. Keeping this in the layout rather than in the
    initialiser means a new tensor cannot be added without saying how it starts.
    """

    LINEAR = "linear"
    EMBEDDING = "embedding"
    NORM = "norm"


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: tuple[int, ...]
    kind: Kind
    # Which expert this belongs to, if any. None means every holder keeps it.
    expert: int | None = None

    @property
    def numel(self) -> int:
        return prod(self.shape)

    @property
    def fan_in(self) -> int:
        """The dimension a uniform initialisation is scaled by.

        For a linear weight stored as (out, in) that is the trailing dimension;
        for an embedding table stored as (vocab, d_model) it is also the
        trailing one, which is why both take the same rule and neither needs a
        special case.
        """
        return self.shape[-1]


def _layer(spec: ModelSpec, i: int) -> list[TensorSpec]:
    d, hd = spec.d_model, spec.head_dim
    p = f"layers.{i}"
    out = [
        TensorSpec(f"{p}.attn_norm.weight", (d,), Kind.NORM),
        TensorSpec(f"{p}.attn.wq.weight", (spec.n_heads * hd, d), Kind.LINEAR),
        TensorSpec(f"{p}.attn.wk.weight", (spec.n_kv_heads * hd, d), Kind.LINEAR),
        TensorSpec(f"{p}.attn.wv.weight", (spec.n_kv_heads * hd, d), Kind.LINEAR),
        TensorSpec(f"{p}.attn.wo.weight", (d, spec.n_heads * hd), Kind.LINEAR),
    ]
    if spec.qk_norm:
        out.append(TensorSpec(f"{p}.attn.q_norm.weight", (hd,), Kind.NORM))
        out.append(TensorSpec(f"{p}.attn.k_norm.weight", (hd,), Kind.NORM))
    out.append(TensorSpec(f"{p}.ffn_norm.weight", (d,), Kind.NORM))

    if not spec.layer_is_moe(i):
        out += [
            TensorSpec(f"{p}.ffn.gate.weight", (spec.d_ff, d), Kind.LINEAR),
            TensorSpec(f"{p}.ffn.up.weight", (spec.d_ff, d), Kind.LINEAR),
            TensorSpec(f"{p}.ffn.down.weight", (d, spec.d_ff), Kind.LINEAR),
        ]
        return out

    # The router is held by every participant, sharded or not: routing decides
    # which expert a token goes to, so a holder that could not compute it could
    # not tell whether the token was even its business.
    out.append(TensorSpec(f"{p}.moe.router.weight", (spec.n_experts, d), Kind.LINEAR))
    de = spec.d_ff_expert
    for e in range(spec.n_experts):
        q = f"{p}.moe.experts.{e}"
        out += [
            TensorSpec(f"{q}.gate.weight", (de, d), Kind.LINEAR, expert=e),
            TensorSpec(f"{q}.up.weight", (de, d), Kind.LINEAR, expert=e),
            TensorSpec(f"{q}.down.weight", (d, de), Kind.LINEAR, expert=e),
        ]
    # Shared experts run for every token, so like the router they are held by
    # everyone and carry no expert index.
    for s in range(spec.n_shared_experts):
        q = f"{p}.moe.shared.{s}"
        out += [
            TensorSpec(f"{q}.gate.weight", (de, d), Kind.LINEAR),
            TensorSpec(f"{q}.up.weight", (de, d), Kind.LINEAR),
            TensorSpec(f"{q}.down.weight", (d, de), Kind.LINEAR),
        ]
    return out


def layout(spec: ModelSpec) -> list[TensorSpec]:
    """Every tensor in the model, in the one order the network agrees on."""
    out = [TensorSpec("embed.weight", (spec.vocab_size, spec.d_model), Kind.EMBEDDING)]
    for i in range(spec.n_layers):
        out += _layer(spec, i)
    out.append(TensorSpec("norm.weight", (spec.d_model,), Kind.NORM))
    if not spec.tie_embeddings:
        out.append(TensorSpec("lm_head.weight", (spec.vocab_size, spec.d_model),
                              Kind.EMBEDDING))
    return out


def parameter_count(spec: ModelSpec) -> int:
    """Counted from the tensors rather than from the formula.

    Two independent descriptions of one model, which is the point: they are
    compared by ModelTests.TheFormulaMatchesTheTensors, and a mistake in either
    shows up as a disagreement rather than as a model that quietly has the
    wrong number of weights.
    """
    return sum(t.numel for t in layout(spec))


def shard_of(spec: ModelSpec, tensor: TensorSpec) -> int | None:
    """Which shard holds this tensor, or None if every holder does.

    Shared parameters — embeddings, attention, norms, routers, shared experts —
    are held by everyone, because a holder that lacked them could not run a
    forward pass at all. Only routed experts divide.
    """
    if tensor.expert is None:
        return None
    return tensor.expert // spec.experts_per_shard


def shard_layout(spec: ModelSpec, shard: int) -> list[TensorSpec]:
    """What one holder stores: everything shared, plus its own experts."""
    if not 0 <= shard < spec.expert_shard_count:
        raise ValueError(f"layout: shard {shard} of {spec.expert_shard_count}")
    return [t for t in layout(spec) if shard_of(spec, t) in (None, shard)]


def shard_parameter_count(spec: ModelSpec, shard: int = 0) -> int:
    return sum(t.numel for t in shard_layout(spec, shard))


def active_parameter_count(spec: ModelSpec) -> int:
    """Parameters a single token multiplies against.

    Counted from the tensors, so it can be checked against the formula the same
    way the total is. A token reaches every shared tensor, the router, the
    shared experts, and exactly `n_experts_active` of the routed ones — which
    of them depends on the token, but how many does not.
    """
    total = 0
    for t in layout(spec):
        if t.expert is None:
            total += t.numel
    if not spec.is_moe:
        return total
    # Every routed expert has the same shape, so the active share is a count.
    per_expert = 3 * spec.d_model * spec.d_ff_expert
    return total + spec.n_moe_layers * spec.n_experts_active * per_expert
