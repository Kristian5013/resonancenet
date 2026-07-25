"""Llama-style RN model (GQA + SwiGLU + RMSNorm + RoPE + optional QK-norm),
llama.cpp/GGUF-native. Built minimal-but-faithful so the memory envelope of the
real architecture can be measured before the training harness is written."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.type_as(self.weight) * self.weight)


def precompute_rope(head_dim, max_seq, theta):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq).float()
    freqs = torch.outer(t, inv)                       # (seq, head_dim/2)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x, cos, sin):
    # x: (B, H, T, D)
    B, H, T, D = x.shape
    x1, x2 = x[..., : D // 2], x[..., D // 2:]
    cos = cos[:T].view(1, 1, T, D // 2)
    sin = sin[:T].view(1, 1, T, D // 2)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.nh, self.nkv = cfg.n_heads, cfg.n_kv_heads
        self.hd = cfg.d_model // cfg.n_heads
        self.rep = self.nh // self.nkv
        self.wq = nn.Linear(cfg.d_model, self.nh * self.hd, bias=False)
        self.wk = nn.Linear(cfg.d_model, self.nkv * self.hd, bias=False)
        self.wv = nn.Linear(cfg.d_model, self.nkv * self.hd, bias=False)
        self.wo = nn.Linear(self.nh * self.hd, cfg.d_model, bias=False)
        self.qk_norm = cfg.qk_norm
        if self.qk_norm:
            self.q_norm = RMSNorm(self.hd)
            self.k_norm = RMSNorm(self.hd)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.nh, self.hd).transpose(1, 2)
        k = self.wk(x).view(B, T, self.nkv, self.hd).transpose(1, 2)
        v = self.wv(x).view(B, T, self.nkv, self.hd).transpose(1, 2)
        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        k = k.repeat_interleave(self.rep, dim=1)      # GQA expand
        v = v.repeat_interleave(self.rep, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(y)


class SwiGLU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.w_gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w_up = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w_down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.ln2 = RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.ln1(x), cos, sin)
        x = x + self.mlp(self.ln2(x))
        return x


class LlamaModel(nn.Module):
    def __init__(self, cfg, grad_checkpointing=True):
        super().__init__()
        self.cfg = cfg
        self.grad_checkpointing = grad_checkpointing
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.tok.weight = self.head.weight
        hd = cfg.d_model // cfg.n_heads
        cos, sin = precompute_rope(hd, cfg.ctx_train, cfg.rope_theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, idx, targets=None):
        x = self.tok(idx)
        for blk in self.blocks:
            if self.grad_checkpointing and self.training:
                x = checkpoint(blk, x, self.cos, self.sin, use_reentrant=False)
            else:
                x = blk(x, self.cos, self.sin)
        x = self.norm(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def num_params(self):
        n = sum(p.numel() for p in self.parameters())
        if self.cfg.tie_embeddings:      # tok/head shared -> counted once already
            pass
        return n
