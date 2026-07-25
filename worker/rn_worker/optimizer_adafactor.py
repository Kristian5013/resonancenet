"""Layerwise Adafactor — the single-24GB-card training recipe (measured).

Adafactor's factored second-moment states are tiny and need no SVD (GaLore was
rejected: SVD too slow at 1-7B). Applied LAYERWISE: one optimizer per parameter,
stepped in a post-accumulate-grad hook so gradients are freed one parameter at a
time — that is what keeps a full-parameter update inside 24 GB.

Consequence: the step happens during backward, once per micro-batch. Gradient
accumulation (K micro-batches -> 1 step) is therefore NOT supported here; the
inner loop is micro_batch=1, step-per-backward. Larger effective batch is the
DiLoCo layer's job (accumulate pseudo-gradients across the outer round), not this
optimizer's. If per-step accumulation is later needed, the hook must buffer.
"""

import torch
from transformers.optimization import Adafactor


class LayerwiseAdafactor:
    def __init__(self, model: torch.nn.Module, lr: float):
        self._opt = {}
        self._handles = []
        for p in model.parameters():
            if not p.requires_grad:
                continue
            self._opt[p] = Adafactor([p], lr=lr, relative_step=False,
                                     scale_parameter=False, warmup_init=False)
            self._handles.append(p.register_post_accumulate_grad_hook(self._step_and_free))

    def _step_and_free(self, p):
        opt = self._opt[p]
        opt.step()
        opt.zero_grad(set_to_none=True)

    def set_lr(self, lr: float):
        # Must be called BEFORE backward: the per-param step fires inside backward
        # and reads the current lr.
        for opt in self._opt.values():
            for group in opt.param_groups:
                group["lr"] = lr

    def remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()
