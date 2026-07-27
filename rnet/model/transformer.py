"""The model itself — dense or mixture, in the arithmetic the round pinned.

THE PARAMETER NAMES ARE THE LAYOUT'S NAMES. Every module here is built so that
`state_dict()` keys equal `layout(spec)` names exactly, and a test asserts it.
That is not tidiness: the weights hash is a root over tensors in layout order,
so a model whose parameters are named or ordered differently would hash to
something else while holding identical numbers. Making the names the same thing
rather than two things kept in agreement removes the failure entirely.

THE ARITHMETIC IS NOT THIS MODULE'S CHOICE. dtypes and the attention kernel
arrive in a Numerics, which came out of a hashed artifact. Nothing here decides
to use flash because it is faster or fp32 because it is safer — a worker that
picked its own would compute an update nobody could reproduce and would be
indistinguishable from one that cheated.

WHAT IS DELIBERATELY ABSENT. There is no KV cache and no generation path. This
model exists to be trained and to be replayed bit-for-bit by a verifier; caching
is an inference optimisation whose interaction with determinism would have to be
argued rather than assumed. Export to GGUF is how a trained model gets run.
"""

from __future__ import annotations

import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..consensus.model_spec import ModelSpec
from ..consensus.numerics import AttentionKernel, DType, Numerics


def torch_dtype(d: DType) -> torch.dtype:
    """The torch dtype a consensus dtype names.

    Looked up by name so `numerics` stays importable without torch — a node that
    only verifies consensus must not need a deep-learning stack, and today it
    does not.
    """
    dtype = getattr(torch, d.torch_name, None)
    if dtype is None:
        raise RuntimeError(
            f"model: this torch build has no {d.torch_name}; the round needs it")
    return dtype


@contextlib.contextmanager
def attention_backend(kernel: AttentionKernel):
    """Pin the attention implementation for the duration of a forward pass.

    Flash and math are each individually reproducible and are NOT bit-identical
    to each other. Letting torch choose by heuristic means the same weights and
    the same batch give different updates on different machines, which is the
    exact failure the whole verification scheme is built to detect — so it must
    not be produced by accident.
    """
    from torch.nn.attention import SDPBackend, sdpa_kernel
    backend = {AttentionKernel.FLASH: SDPBackend.FLASH_ATTENTION,
               AttentionKernel.MATH: SDPBackend.MATH}[kernel]
    with sdpa_kernel(backend):
        yield


class RMSNorm(nn.Module):
    """Root-mean-square norm, reduced in the accumulation dtype.

    The reduction is the part that matters for reproducibility: summing 3072
    squares in bf16 gives a different answer depending on the order the kernel
    happens to batch them, so it is done in the round's accumulation dtype and
    narrowed once at the end.
    """

    def __init__(self, dim: int, accum: torch.dtype, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.accum = accum
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x.to(self.accum)
        out = out * torch.rsqrt(out.pow(2).mean(-1, keepdim=True) + self.eps)
        return (out * self.weight.to(self.accum)).to(x.dtype)


def rope_frequencies(head_dim: int, seq_len: int, theta: float,
                     device, accum: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Cosine and sine tables for rotary embeddings.

    Built in the accumulation dtype and kept there: these are a fixed function
    of position, computed once, and rounding them to bf16 would put a
    position-dependent error into every attention score for no saving worth
    having.
    """
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device,
                                        dtype=torch.float32) / head_dim))
    pos = torch.arange(seq_len, device=device, dtype=torch.float32)
    angles = torch.outer(pos, inv)
    return angles.cos().to(accum), angles.sin().to(accum)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x is (batch, heads, seq, head_dim); the last dimension rotates in pairs."""
    a, b = x[..., 0::2], x[..., 1::2]
    c = cos[: x.shape[-2]].unsqueeze(0).unsqueeze(0).to(x.dtype)
    s = sin[: x.shape[-2]].unsqueeze(0).unsqueeze(0).to(x.dtype)
    return torch.stack((a * c - b * s, a * s + b * c), dim=-1).flatten(-2)


class Attention(nn.Module):
    """Grouped-query attention with optional query/key norms."""

    def __init__(self, spec: ModelSpec, numerics: Numerics):
        super().__init__()
        self.spec, self.numerics = spec, numerics
        d, hd = spec.d_model, spec.head_dim
        self.wq = nn.Linear(d, spec.n_heads * hd, bias=False)
        self.wk = nn.Linear(d, spec.n_kv_heads * hd, bias=False)
        self.wv = nn.Linear(d, spec.n_kv_heads * hd, bias=False)
        self.wo = nn.Linear(spec.n_heads * hd, d, bias=False)
        accum = torch_dtype(numerics.accum_dtype)
        if spec.qk_norm:
            self.q_norm = RMSNorm(hd, accum)
            self.k_norm = RMSNorm(hd, accum)

    def forward(self, x, cos, sin):
        b, t, _ = x.shape
        spec = self.spec
        hd, nh, nkv = spec.head_dim, spec.n_heads, spec.n_kv_heads

        q = self.wq(x).view(b, t, nh, hd).transpose(1, 2)
        k = self.wk(x).view(b, t, nkv, hd).transpose(1, 2)
        v = self.wv(x).view(b, t, nkv, hd).transpose(1, 2)
        if spec.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        if nkv != nh:
            # Repeat rather than expand: the fused kernels want contiguous heads,
            # and enable_gqa is not available on every torch this must run on.
            k = k.repeat_interleave(nh // nkv, dim=1)
            v = v.repeat_interleave(nh // nkv, dim=1)

        with attention_backend(self.numerics.attention_kernel):
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.wo(out.transpose(1, 2).reshape(b, t, nh * hd))


class SwiGLU(nn.Module):
    """The dense feed-forward block: gate, up, down."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class MoE(nn.Module):
    """Top-k routed experts, plus any shared experts that run for every token.

    The routing is computed in the accumulation dtype and the top-k is taken
    there. That matters more than it looks: in bf16 two experts' scores collide
    often enough that which one wins would depend on the reduction order, and a
    token routed differently on two machines produces a different update from
    identical weights.

    THE LOOP IS OVER EXPERTS, NOT TOKENS. Each expert processes the tokens that
    chose it in one batched call. A loop over tokens would be correct and
    unusably slow; a dense computation of every expert for every token would be
    correct and would throw away the entire reason to use a mixture.
    """

    def __init__(self, spec: ModelSpec, numerics: Numerics):
        super().__init__()
        self.spec = spec
        self.accum = torch_dtype(numerics.accum_dtype)
        self.router = nn.Linear(spec.d_model, spec.n_experts, bias=False)
        self.experts = nn.ModuleList(
            SwiGLU(spec.d_model, spec.d_ff_expert) for _ in range(spec.n_experts))
        self.shared = nn.ModuleList(
            SwiGLU(spec.d_model, spec.d_ff_expert) for _ in range(spec.n_shared_experts))

    def forward(self, x):
        b, t, d = x.shape
        flat = x.reshape(-1, d)

        logits = self.router(flat).to(self.accum)
        weights, chosen = torch.topk(logits, self.spec.n_experts_active, dim=-1)
        weights = torch.softmax(weights, dim=-1).to(x.dtype)

        out = torch.zeros_like(flat)
        for e, expert in enumerate(self.experts):
            rows, slot = torch.where(chosen == e)
            if rows.numel() == 0:
                continue
            # index_add_ rather than out[rows] += ... . Today the two are
            # equivalent: topk returns distinct indices, so a row selects a
            # given expert at most once and `rows` holds no duplicates —
            # asserted by MoETests.ARowSelectsAnExpertAtMostOnce. index_add_ is
            # used because it does not DEPEND on that. If routing ever allows a
            # token to weight one expert twice, plain indexing would silently
            # keep the last write instead of the sum, and the symptom would be a
            # slightly wrong model rather than an error.
            out.index_add_(0, rows,
                           expert(flat[rows]) * weights[rows, slot].unsqueeze(-1))
        for expert in self.shared:
            out = out + expert(flat)
        return out.view(b, t, d)


class Block(nn.Module):
    def __init__(self, spec: ModelSpec, numerics: Numerics, index: int):
        super().__init__()
        accum = torch_dtype(numerics.accum_dtype)
        self.attn_norm = RMSNorm(spec.d_model, accum)
        self.attn = Attention(spec, numerics)
        self.ffn_norm = RMSNorm(spec.d_model, accum)
        if spec.layer_is_moe(index):
            self.moe = MoE(spec, numerics)
        else:
            self.ffn = SwiGLU(spec.d_model, spec.d_ff)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        feed = self.moe if hasattr(self, "moe") else self.ffn
        return x + feed(self.ffn_norm(x))


class Transformer(nn.Module):
    """The whole model, named exactly as the canonical layout names it."""

    def __init__(self, spec: ModelSpec, numerics: Numerics,
                 grad_checkpointing: bool = True):
        super().__init__()
        spec.validate()
        numerics.validate()
        self.spec, self.numerics = spec, numerics
        self.grad_checkpointing = grad_checkpointing

        self.embed = nn.Embedding(spec.vocab_size, spec.d_model)
        self.layers = nn.ModuleList(Block(spec, numerics, i) for i in range(spec.n_layers))
        self.norm = RMSNorm(spec.d_model, torch_dtype(numerics.accum_dtype))
        if not spec.tie_embeddings:
            self.lm_head = nn.Linear(spec.d_model, spec.vocab_size, bias=False)

        self._cos = self._sin = None

    def _rope(self, device):
        if self._cos is None or self._cos.device != device:
            self._cos, self._sin = rope_frequencies(
                self.spec.head_dim, self.spec.seq_len, float(self.spec.rope_theta),
                device, torch_dtype(self.numerics.accum_dtype))
        return self._cos, self._sin

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        cos, sin = self._rope(tokens.device)
        x = self.embed(tokens)
        for layer in self.layers:
            if self.grad_checkpointing and self.training:
                # Activation checkpointing is what puts a 16384-token context
                # inside 24 GB. use_reentrant=False because the reentrant path
                # does not compose with the sdpa backend context manager above.
                x = torch.utils.checkpoint.checkpoint(
                    layer, x, cos, sin, use_reentrant=False)
            else:
                x = layer(x, cos, sin)
        x = self.norm(x)
        if self.spec.tie_embeddings:
            return F.linear(x, self.embed.weight)
        return self.lm_head(x)

    def loss(self, tokens: torch.Tensor) -> torch.Tensor:
        """Next-token cross entropy over a window of `seq_len + 1` tokens.

        The window carries one extra token so inputs and targets come from the
        same draw; splitting here rather than at the caller keeps the two from
        ever being off by one relative to each other.
        """
        if tokens.dim() != 2 or tokens.shape[1] < 2:
            # Otherwise the split leaves nothing on either side and cross entropy
            # over an empty batch returns NaN — a loss that looks like a training
            # failure rather than like the malformed input it is.
            raise ValueError(
                f"model: a loss window needs at least 2 tokens, got shape "
                f"{tuple(tokens.shape)}")
        logits = self(tokens[:, :-1])
        return F.cross_entropy(
            logits.reshape(-1, self.spec.vocab_size).float(),
            tokens[:, 1:].reshape(-1))
