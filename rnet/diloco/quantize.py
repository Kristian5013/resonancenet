"""Turning a worker's pseudo-gradient into bytes the network can add up.

A contribution is the difference between the weights a worker started with and
the weights it ended with, quantised to a small integer per parameter with one
shared scale. Three properties, and each of them is load-bearing:

  * THE SCALE IS A POWER OF TWO. Applying it touches only a float's exponent, so
    dequantising is exact and never rounds. A decimal scale would introduce a
    rounding whose result depends on the order operations happen in, and the
    whole verification scheme rests on that not being true.

  * THE RANGE IS SYMMETRIC ABOUT ZERO. Two's complement offers one more negative
    value than positive; using it would bias the mean of every update in a
    direction nobody chose, forever, by about half a quantisation step.

  * TIES ROUND AWAY FROM ZERO, NOT TO EVEN. numpy's default is to even, which is
    right for repeated arithmetic and wrong here: it is a second rule to agree
    on, and the one that matters is that every implementation applies the SAME
    rule. Away-from-zero is the one stated, and it is stated because "whatever
    numpy does by default" is not a specification.

THE EXPONENT IS CHOSEN WITHOUT A LOGARITHM. `math.frexp` reads the exponent out
of the float's bits; `log2` computes it, and different C libraries compute it
differently in the last place. A worker whose scale differed by one from another
worker's would produce a contribution that dequantised to something twice as
large, which is not a rounding difference but a wrong answer.

see: docs/diloco.md
"""

from __future__ import annotations

import math

import numpy as np

from ..consensus.numerics import ContributionFormat

# A scale beyond this is not a number a training run produces. Bounding it stops
# a contribution whose exponent is absurd from turning into infinity or zero on
# dequantisation, where it would poison the aggregate of everyone who added it.
MAX_SCALE_EXP = 300


class QuantiseError(Exception):
    pass


def choose_scale_exp(values: np.ndarray, fmt: ContributionFormat) -> int:
    """The smallest exponent that fits `values` inside the format's range.

    Smallest, because a larger one throws away precision for nothing: the whole
    point of a per-contribution scale is to spend the available integers on the
    range the values actually occupy.
    """
    if values.size == 0:
        raise QuantiseError("quantise: an empty update is not an update")
    finite = np.isfinite(values)
    if not finite.all():
        # A non-finite pseudo-gradient means the inner loop diverged. Encoding it
        # would spread one worker's blown-up run to everyone who aggregates it.
        raise QuantiseError(
            f"quantise: {int((~finite).sum())} of {values.size} values are not finite")

    max_abs = float(np.abs(values).max())
    if max_abs == 0.0:
        # A worker that changed nothing still has to say so, and 0 is the
        # exponent that makes every zero dequantise to zero.
        return 0

    limit = fmt.max_magnitude
    _, k = math.frexp(max_abs)          # max_abs = m * 2**k, 0.5 <= m < 1
    exp = k - (fmt.value - 1)
    # frexp's mantissa can be close enough to 1 that the shift above still
    # overflows the range by one step. Checked rather than reasoned about.
    if math.ldexp(max_abs, -exp) > limit:
        exp += 1
    if abs(exp) > MAX_SCALE_EXP:
        raise QuantiseError(f"quantise: scale exponent {exp} is out of range")
    return exp


def quantize(values: np.ndarray, exp: int, fmt: ContributionFormat) -> np.ndarray:
    """Scale by 2**-exp, round ties away from zero, clamp to the format.

    Clamping is not expected to bite when `exp` came from `choose_scale_exp`,
    and it is here because a worker chooses its own exponent: one that is too
    small must produce a saturated contribution rather than integers that wrap.
    """
    if abs(exp) > MAX_SCALE_EXP:
        raise QuantiseError(f"quantise: scale exponent {exp} is out of range")
    scaled = np.ldexp(np.asarray(values, dtype=np.float64), -exp)
    if not np.isfinite(scaled).all():
        raise QuantiseError("quantise: scaling produced a non-finite value")
    # sign * floor(|x| + 0.5) is round-half-away-from-zero, stated rather than
    # inherited: np.round would take ties to even.
    rounded = np.sign(scaled) * np.floor(np.abs(scaled) + 0.5)
    limit = fmt.max_magnitude
    return np.clip(rounded, -limit, limit).astype(np.int64)


def clamped_count(values: np.ndarray, exp: int, fmt: ContributionFormat) -> int:
    """How many values the format could not hold at this exponent.

    Worth reporting rather than hiding: a contribution that saturates is one
    whose largest updates were flattened, and a worker seeing this rise knows
    its exponent is wrong before its contributions start being useless.
    """
    scaled = np.abs(np.ldexp(np.asarray(values, dtype=np.float64), -exp))
    return int((np.floor(scaled + 0.5) > fmt.max_magnitude).sum())


def dequantize(q: np.ndarray, exp: int) -> np.ndarray:
    """Integers back to floats. Exact, because the scale is a power of two."""
    return np.ldexp(np.asarray(q, dtype=np.float64), exp)


def quantize_update(values: np.ndarray, fmt: ContributionFormat
                    ) -> tuple[np.ndarray, int]:
    """The whole worker-side operation: pick an exponent and encode."""
    exp = choose_scale_exp(values, fmt)
    return quantize(values, exp, fmt), exp


def pack(q: np.ndarray, fmt: ContributionFormat) -> bytes:
    """Quantised values as bytes on the wire.

    int8 is one byte each. The narrower formats pack two values per byte for
    int4, and int6 is stored one per byte — six bits packed four-to-three-bytes
    would save a quarter of the payload and cost a bit-shuffling step that has to
    be identical everywhere, which is a poor trade for a format whose reason to
    exist is bandwidth on a link that is already not the bottleneck.
    """
    if fmt is ContributionFormat.INT4_POW2:
        if q.size % 2:
            raise QuantiseError("quantise: int4 packing needs an even count")
        # Low nibble first, so a hex dump reads in element order.
        lo = (q[0::2].astype(np.int64) & 0x0F).astype(np.uint8)
        hi = (q[1::2].astype(np.int64) & 0x0F).astype(np.uint8)
        return (lo | (hi << 4)).tobytes()
    return q.astype(np.int8).tobytes()


def unpack(raw: bytes, count: int, fmt: ContributionFormat) -> np.ndarray:
    if fmt is ContributionFormat.INT4_POW2:
        if count % 2:
            raise QuantiseError("quantise: int4 unpacking needs an even count")
        if len(raw) != count // 2:
            raise QuantiseError(
                f"quantise: {len(raw)} bytes hold {len(raw) * 2} int4 values, "
                f"{count} wanted")
        packed = np.frombuffer(raw, dtype=np.uint8)
        out = np.empty(count, dtype=np.int64)
        # Sign-extend four bits: values 8..15 are -8..-1.
        out[0::2] = (packed & 0x0F).astype(np.int64)
        out[1::2] = (packed >> 4).astype(np.int64)
        return np.where(out > 7, out - 16, out)
    if len(raw) != count:
        raise QuantiseError(f"quantise: {len(raw)} bytes, {count} values wanted")
    return np.frombuffer(raw, dtype=np.int8).astype(np.int64)
