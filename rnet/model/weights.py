"""Moving weights between the canonical bytes and a live model.

The canonical form is big-endian bfloat16 in layout order — the same bytes the
initial weights are derived as, the same bytes a checkpoint hashes. A model in
memory is little-endian torch tensors in whatever dtype the round pinned. This
module is the one place that converts, so the byte order lives in a single
function instead of at every call site that ever writes a weight to disk.

WHY BIG-ENDIAN FOR SOMETHING THAT ONLY EVER RUNS ON LITTLE-ENDIAN MACHINES.
Because "whatever this machine does" is not a specification. Every integer this
project puts on a wire is big-endian; making weights the exception would mean
the hash of a model depends on the architecture that computed it, and the first
person to notice would be whoever ran it on the wrong one.
"""

from __future__ import annotations

import numpy as np
import torch

from ..consensus.model_spec import ModelSpec
from ..model.layout import layout, shard_layout
from .transformer import Transformer


def bf16_words_to_tensor(words: np.ndarray, shape: tuple[int, ...],
                         dtype: torch.dtype, device) -> torch.Tensor:
    """Raw bf16 words into a tensor of the round's parameter dtype.

    Through bfloat16 always, even when the target is float32: the canonical
    value IS a bf16, and widening it afterwards is exact, whereas reconstructing
    a float32 from the same bits by another route would not have to be.
    """
    bits = torch.from_numpy(np.ascontiguousarray(words, dtype=np.uint16))
    return bits.view(torch.bfloat16).reshape(shape).to(device=device, dtype=dtype)


def tensor_to_bf16_words(t: torch.Tensor) -> np.ndarray:
    """A tensor back to raw bf16 words, rounding once if it is wider.

    torch's own float32 -> bfloat16 conversion rounds half to even, which is
    what the derivation does, so the two agree. Proved by
    WeightsTests.LoadingThenSavingIsTheIdentity.
    """
    return (t.detach().to(torch.bfloat16).contiguous()
             .view(torch.uint16).cpu().numpy().reshape(-1).copy())


def load_weights(model: Transformer, weights: dict[str, np.ndarray], *,
                 shard: int | None = None) -> None:
    """Put canonical weights into a model, insisting the two agree exactly.

    Every tensor the layout names must be present and the right size, and
    nothing else may be. A loader that skipped a missing tensor would leave it
    at whatever the constructor produced, and the model would train from weights
    the network never agreed on while reporting the right hash for the rest.
    """
    spec = model.spec
    expected = layout(spec) if shard is None else shard_layout(spec, shard)
    names = {t.name for t in expected}

    extra = set(weights) - names
    if extra:
        raise KeyError(f"weights: {len(extra)} tensor(s) the layout does not name, "
                       f"first is {sorted(extra)[0]}")

    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device
    params = dict(model.named_parameters())

    with torch.no_grad():
        for t in expected:
            words = weights.get(t.name)
            if words is None:
                raise KeyError(f"weights: {t.name} is missing")
            if words.size != t.numel:
                raise ValueError(
                    f"weights: {t.name} has {words.size} elements, the layout says "
                    f"{t.numel}")
            param = params.get(t.name)
            if param is None:
                raise KeyError(
                    f"weights: the model has no parameter named {t.name}; the module "
                    f"tree and the canonical layout have diverged")
            param.copy_(bf16_words_to_tensor(words, t.shape, dtype, device))


def save_weights(model: Transformer, *, shard: int | None = None
                 ) -> dict[str, np.ndarray]:
    """A model's weights as canonical words, ready to hash or to write."""
    spec = model.spec
    wanted = layout(spec) if shard is None else shard_layout(spec, shard)
    params = dict(model.named_parameters())
    out = {}
    for t in wanted:
        param = params.get(t.name)
        if param is None:
            raise KeyError(f"weights: the model has no parameter named {t.name}")
        out[t.name] = tensor_to_bf16_words(param)
    return out


def weights_hash(model: Transformer) -> bytes:
    """What this model would be committed to as.

    The number a worker reports after applying an update, and what a node checks
    a received checkpoint against.
    """
    from ..consensus.init import hash_of_values
    return hash_of_values(model.spec, save_weights(model))


def build(spec: ModelSpec, numerics, genesis_hash: bytes, *, device="cpu",
          grad_checkpointing: bool = True, shard: int | None = None) -> Transformer:
    """A model at the network's initial weights.

    The whole loop in one call: derive from the genesis hash, build the module
    tree, load. What comes back hashes to the network's weights anchor, which is
    what makes "the network started here" a checkable claim rather than a
    statement about a file somebody uploaded.
    """
    from ..consensus.init import derive_all
    model = Transformer(spec, numerics, grad_checkpointing=grad_checkpointing)
    model = model.to(device=device, dtype=torch_param_dtype(numerics))
    load_weights(model, derive_all(spec, genesis_hash, shard=shard), shard=shard)
    return model


def torch_param_dtype(numerics) -> torch.dtype:
    from .transformer import torch_dtype
    return torch_dtype(numerics.param_dtype)
