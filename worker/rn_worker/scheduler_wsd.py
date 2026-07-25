"""WSD (warmup-stable-decay) learning-rate schedule.

Chosen because it is the one schedule that fits continuous learning: hold a
stable plateau, publish a checkpoint after a short decay, then resume from the
plateau — no cosine curve to restart. Equivalent quality to cosine.
"""


class WSD:
    def __init__(self, peak_lr: float, total_steps: int, warmup_steps: int,
                 decay_frac: float = 0.2, min_lr: float = 0.0):
        self.peak = peak_lr
        self.total = max(1, total_steps)
        self.warmup = max(0, warmup_steps)
        self.decay_steps = max(1, int(decay_frac * total_steps))
        self.min_lr = min_lr

    def lr_at(self, step: int) -> float:
        if step < self.warmup:                                   # linear warmup
            return self.peak * (step + 1) / max(1, self.warmup)
        decay_start = self.total - self.decay_steps
        if step < decay_start:                                   # stable plateau
            return self.peak
        t = (step - decay_start) / self.decay_steps              # linear decay
        t = min(1.0, max(0.0, t))
        return self.peak + (self.min_lr - self.peak) * t
