"""Python mirror of src/diloco — quantisation, aggregation, outer optimizer.

Every function here must agree with the C++ authority to the last bit. A worker
that quantised differently would submit a payload no verifier could reproduce,
and its honest work would look like fraud.

The arithmetic is integer and the quantisation scale is a power of two, which is
exactly why cross-language agreement is achievable at all: scaling by 2^k only
touches the exponent field of a float, so no rounding difference can creep in.
"""

from __future__ import annotations

import math

from . import canon

QUANT_MAX = 127
QUANT_MIN = -127
MAX_SCALE_EXP = 64
MIN_SCALE_EXP = -64
MAX_ALIGN_SHIFT = 24
Q16_ONE = 1 << 16


class DilocoError(Exception):
    pass


# ---------------------------------------------------------------------------
# Quantisation
# ---------------------------------------------------------------------------
def round_ties_away_from_zero(x: float) -> int:
    """Python's round() is banker's rounding; the protocol rounds halves away
    from zero, so this cannot use the builtin."""
    return math.floor(x + 0.5) if x >= 0 else math.ceil(x - 0.5)


def choose_scale_exp(deltas) -> int:
    if len(deltas) == 0:
        raise DilocoError("quantize: empty delta vector")
    max_abs = 0.0
    for v in deltas:
        if not math.isfinite(v):
            # A non-finite delta means the local run diverged; quantising it would
            # turn a NaN into a plausible-looking integer.
            raise DilocoError("quantize: non-finite value in pseudo-gradient")
        max_abs = max(max_abs, abs(float(v)))
    if max_abs == 0.0:
        return 0

    scale_exp = 0
    while max_abs * (2.0**scale_exp) > QUANT_MAX:
        scale_exp -= 1
    while max_abs * (2.0 ** (scale_exp + 1)) <= QUANT_MAX:
        scale_exp += 1
    return max(MIN_SCALE_EXP, min(MAX_SCALE_EXP, scale_exp))


def quantize(deltas, scale_exp: int) -> tuple[list[int], int]:
    """Returns (values, clamped_count). Clamping is reported, never silent: it is
    how a diverging worker becomes visible."""
    if not (MIN_SCALE_EXP <= scale_exp <= MAX_SCALE_EXP):
        raise DilocoError(f"quantize: scale_exp {scale_exp} out of range")
    scale = 2.0**scale_exp
    values, clamped = [], 0
    for v in deltas:
        if not math.isfinite(v):
            raise DilocoError("quantize: non-finite value in pseudo-gradient")
        q = round_ties_away_from_zero(float(v) * scale)
        if q > QUANT_MAX:
            q, clamped = QUANT_MAX, clamped + 1
        elif q < QUANT_MIN:
            q, clamped = QUANT_MIN, clamped + 1
        values.append(q)
    return values, clamped


def dequantize(values, scale_exp: int) -> list[float]:
    if not (MIN_SCALE_EXP <= scale_exp <= MAX_SCALE_EXP):
        raise DilocoError(f"dequantize: scale_exp {scale_exp} out of range")
    inv = 2.0 ** (-scale_exp)
    return [float(q) * inv for q in values]


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def rounded_div(numerator: int, denominator: int) -> int:
    """Integer division rounding halves away from zero, symmetric about zero so
    repeated rounds cannot drift in one direction."""
    if denominator == 0:
        return 0
    negative = (numerator < 0) != (denominator < 0)
    n, d = abs(numerator), abs(denominator)
    q = (n + d // 2) // d
    return -q if negative else q


def sum_contributions(contributions, n_params: int) -> tuple[list[int], int]:
    """contributions: iterable of (worker_id, scale_exp, values).
    Returns (summed values, target scale_exp).

    Every contribution is included — this is a sum, not a selection. Integer
    addition is associative, so arrival order cannot change the result.
    """
    contributions = list(contributions)
    if not contributions:
        raise DilocoError("aggregate: no contributions")

    target_exp = MIN_SCALE_EXP - 1
    for worker_id, scale_exp, values in contributions:
        if len(values) != n_params:
            raise DilocoError(
                f"aggregate: worker {worker_id} sent {len(values)} values, expected {n_params}"
            )
        if not (MIN_SCALE_EXP <= scale_exp <= MAX_SCALE_EXP):
            raise DilocoError(f"aggregate: worker {worker_id} scale_exp out of range")
        target_exp = max(target_exp, scale_exp)

    out = [0] * n_params
    for worker_id, scale_exp, values in contributions:
        shift = target_exp - scale_exp
        if shift > MAX_ALIGN_SHIFT:
            raise DilocoError(
                f"aggregate: worker {worker_id} scale differs by 2^{shift} — treating as divergent"
            )
        for i, v in enumerate(values):
            out[i] += v << shift
    return out, target_exp


def average_contributions(contributions, n_params: int) -> tuple[list[int], int, int]:
    contributions = list(contributions)
    values, target_exp = sum_contributions(contributions, n_params)
    count = len(contributions)
    return [rounded_div(v, count) for v in values], target_exp, count


# ---------------------------------------------------------------------------
# Outer optimizer (Nesterov in Q16 fixed point)
# ---------------------------------------------------------------------------
def mul_q16(value: int, q16: int) -> int:
    product = value * q16
    negative = product < 0
    rounded = (abs(product) + (Q16_ONE // 2)) >> 16
    return -rounded if negative else rounded


class OuterOptimizer:
    """Momentum accumulates across rounds, so a last-bit difference would compound
    rather than cancel — hence fixed point rather than float."""

    def __init__(self, n_params: int, scale_exp: int, momentum_q16: int, outer_lr_q16: int,
                 nesterov: bool = True):
        self.n_params = n_params
        self.scale_exp = scale_exp
        self.momentum_q16 = momentum_q16
        self.outer_lr_q16 = outer_lr_q16
        self.nesterov = nesterov
        self.momentum = [0] * n_params
        self.steps_taken = 0

    def step(self, aggregate_values, aggregate_scale_exp: int) -> list[int]:
        if len(aggregate_values) != self.n_params:
            raise DilocoError("outer optimizer: aggregate length mismatch")
        shift = self.scale_exp - aggregate_scale_exp
        if abs(shift) > 32:
            raise DilocoError(f"outer optimizer: aggregate scale differs by 2^{shift}")

        update = []
        for i, g_raw in enumerate(aggregate_values):
            if shift > 0:
                g = g_raw << shift
            elif shift < 0:
                g = rounded_div(g_raw, 1 << (-shift))
            else:
                g = g_raw

            buf = mul_q16(self.momentum[i], self.momentum_q16) + g
            self.momentum[i] = buf
            direction = g + mul_q16(buf, self.momentum_q16) if self.nesterov else buf
            update.append(mul_q16(direction, self.outer_lr_q16))

        self.steps_taken += 1
        return update

    def state_hash(self) -> str:
        w = canon.Writer()
        w.u64(self.n_params)
        w.u32(self.scale_exp & 0xFFFFFFFF)
        w.u64(self.momentum_q16)
        w.u64(self.outer_lr_q16)
        w.u8(1 if self.nesterov else 0)
        w.u64(self.steps_taken)
        for v in self.momentum:
            w.u64(v & 0xFFFFFFFFFFFFFFFF)
        return canon.sha3_256(w.bytes()).hex()


# ---------------------------------------------------------------------------
# Vectorised paths
#
# The functions above are the REFERENCE implementation: scalar, obvious, and the
# thing the cross-language test compares against C++. They are unusable at a
# billion parameters — a Python list of 1e9 ints costs tens of gigabytes.
#
# These numpy versions do the same arithmetic on arrays. They must agree with the
# reference exactly, so `test_vectorised_matches_reference` compares them element
# for element on random inputs; if they ever diverge, a worker and a verifier
# using different paths would disagree about identical work.
# ---------------------------------------------------------------------------
import numpy as _np


def round_ties_away_from_zero_array(x: "_np.ndarray") -> "_np.ndarray":
    """numpy's rint is banker's rounding; the protocol rounds halves away from
    zero. trunc(x + copysign(0.5, x)) is that rule, vectorised."""
    return _np.trunc(x + _np.copysign(0.5, x))


def choose_scale_exp_array(deltas: "_np.ndarray") -> int:
    if deltas.size == 0:
        raise DilocoError("quantize: empty delta vector")
    if not _np.isfinite(deltas).all():
        raise DilocoError("quantize: non-finite value in pseudo-gradient")
    max_abs = float(_np.abs(deltas).max())
    if max_abs == 0.0:
        return 0
    scale_exp = 0
    while max_abs * (2.0**scale_exp) > QUANT_MAX:
        scale_exp -= 1
    while max_abs * (2.0 ** (scale_exp + 1)) <= QUANT_MAX:
        scale_exp += 1
    return max(MIN_SCALE_EXP, min(MAX_SCALE_EXP, scale_exp))


def quantize_array(deltas: "_np.ndarray", scale_exp: int) -> tuple["_np.ndarray", int]:
    if not (MIN_SCALE_EXP <= scale_exp <= MAX_SCALE_EXP):
        raise DilocoError(f"quantize: scale_exp {scale_exp} out of range")
    if not _np.isfinite(deltas).all():
        raise DilocoError("quantize: non-finite value in pseudo-gradient")
    scaled = _np.asarray(deltas, dtype=_np.float64) * (2.0**scale_exp)
    q = round_ties_away_from_zero_array(scaled)
    clamped = int(_np.count_nonzero((q > QUANT_MAX) | (q < QUANT_MIN)))
    return _np.clip(q, QUANT_MIN, QUANT_MAX).astype(_np.int8), clamped


def average_contributions_array(contributions, n_params: int) -> tuple["_np.ndarray", int, int]:
    """contributions: iterable of (worker_id, scale_exp, np.ndarray[int8]).

    Every contribution is summed — nothing is selected. int64 accumulation keeps
    the result exact and independent of arrival order.
    """
    contributions = list(contributions)
    if not contributions:
        raise DilocoError("aggregate: no contributions")

    target_exp = MIN_SCALE_EXP - 1
    for worker_id, scale_exp, values in contributions:
        if len(values) != n_params:
            raise DilocoError(
                f"aggregate: worker {worker_id} sent {len(values)} values, expected {n_params}"
            )
        if not (MIN_SCALE_EXP <= scale_exp <= MAX_SCALE_EXP):
            raise DilocoError(f"aggregate: worker {worker_id} scale_exp out of range")
        target_exp = max(target_exp, scale_exp)

    total = _np.zeros(n_params, dtype=_np.int64)
    for worker_id, scale_exp, values in contributions:
        shift = target_exp - scale_exp
        if shift > MAX_ALIGN_SHIFT:
            raise DilocoError(
                f"aggregate: worker {worker_id} scale differs by 2^{shift} — treating as divergent"
            )
        total += _np.asarray(values, dtype=_np.int64) << shift

    count = len(contributions)
    # Rounded division, halves away from zero, matching rounded_div().
    sign = _np.sign(total)
    averaged = sign * ((_np.abs(total) + count // 2) // count)
    return averaged.astype(_np.int64), target_exp, count


def aggregate_and_step_chunked(contributions, n_params: int, momentum: "_np.ndarray",
                               optimizer_scale_exp: int, momentum_q16: int, outer_lr_q16: int,
                               nesterov: bool, chunk: int = 32_000_000):
    """Aggregate and take the outer step in slices.

    At a billion parameters the whole-array path needs tens of gigabytes: each
    int64 view of a contribution is 8 GB, and there are M of them plus the
    momentum buffer. Slicing bounds the working set regardless of model size,
    which is what lets an ordinary machine run a real round.

    Arithmetic is identical to the whole-array path — the slices are independent,
    so there is nothing for chunking to change.
    """
    contributions = list(contributions)
    if not contributions:
        raise DilocoError("aggregate: no contributions")

    target_exp = MIN_SCALE_EXP - 1
    for worker_id, scale_exp, values in contributions:
        if len(values) != n_params:
            raise DilocoError(
                f"aggregate: worker {worker_id} sent {len(values)} values, expected {n_params}")
        if not (MIN_SCALE_EXP <= scale_exp <= MAX_SCALE_EXP):
            raise DilocoError(f"aggregate: worker {worker_id} scale_exp out of range")
        target_exp = max(target_exp, scale_exp)

    for worker_id, scale_exp, _ in contributions:
        if target_exp - scale_exp > MAX_ALIGN_SHIFT:
            raise DilocoError(
                f"aggregate: worker {worker_id} scale differs by 2^{target_exp - scale_exp} — "
                "treating as divergent")

    count = len(contributions)
    shift_to_optimizer = optimizer_scale_exp - target_exp
    update = _np.empty(n_params, dtype=_np.int64)

    for start in range(0, n_params, chunk):
        stop = min(start + chunk, n_params)
        total = _np.zeros(stop - start, dtype=_np.int64)
        for _, scale_exp, values in contributions:
            total += _np.asarray(values[start:stop], dtype=_np.int64) << (target_exp - scale_exp)

        sign = _np.sign(total)
        averaged = sign * ((_np.abs(total) + count // 2) // count)
        update[start:stop] = outer_step_array(
            momentum[start:stop], averaged, shift_to_optimizer,
            momentum_q16, outer_lr_q16, nesterov)

    return update, target_exp, count


def outer_step_array(momentum: "_np.ndarray", aggregate: "_np.ndarray", shift: int,
                     momentum_q16: int, outer_lr_q16: int, nesterov: bool) -> "_np.ndarray":
    """One Nesterov step in Q16 fixed point, vectorised. Mutates `momentum`."""
    def mul_q16_arr(values, q16):
        product = values * int(q16)
        sign = _np.sign(product)
        return sign * ((_np.abs(product) + (Q16_ONE // 2)) >> 16)

    if shift > 0:
        g = aggregate << shift
    elif shift < 0:
        divisor = 1 << (-shift)
        sign = _np.sign(aggregate)
        g = sign * ((_np.abs(aggregate) + divisor // 2) // divisor)
    else:
        g = aggregate

    momentum[:] = mul_q16_arr(momentum, momentum_q16) + g
    direction = g + mul_q16_arr(momentum, momentum_q16) if nesterov else momentum
    return mul_q16_arr(direction, outer_lr_q16)
