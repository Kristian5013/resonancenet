"""The outer step: momentum over aggregated contributions, in fixed point.

The inner loop is ordinary training and can be as floating-point as it likes,
because a verifier replays it against the same hardware class. The OUTER step is
different: every node applies it, and they must all get the same bytes, so it is
integer arithmetic from end to end.

Q16 THROUGHOUT, BUT ONLY IN THE MULTIPLIERS. Momentum is kept in the same units
as the aggregate — plain integers at the contribution's scale exponent — and the
0.9 and 0.7 are Q16 constants applied to them.

Storing momentum itself in Q16 was the obvious first design, and it fails, but
not where it looks like it would. After a worst-case alignment the aggregate
reaches 127 << 24 and momentum settles around 2.1e10; held in Q16 that is
1.4e15, which fits int64 comfortably. What does not fit is the INTERMEDIATE of
the next multiply: 1.4e15 * 65536 is 9.2e19 against a ceiling of 9.2e18, over by
a factor of ten. The overflow is in an expression, not in a variable, which is
exactly the kind that survives review.

Keeping the fixed point in the constants alone puts the stored value at 2.1e10
against a domain of 1.4e14 — a factor of 6,600. Measured, and the failing
alternative is exercised rather than described, by
OuterTests.TheDomainHoldsAtTheWorstCaseAlignment and
OuterTests.MomentumHeldInQ16WouldOverflowTheMultiply.

WHY NOT JUST USE PYTHON INTEGERS, WHICH CANNOT OVERFLOW. Because the arrays are
numpy int64 and numpy wraps silently. Python's unbounded integers would be
correct and about a hundred times too slow at 400 million parameters — measured
in the implementation this replaces. So the domain is checked explicitly before
every multiply, and a violation raises instead of wrapping into a number that is
wrong but plausible.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

from ..canon.stream import Writer
from .aggregate import AggregateError, rounded_div

Q16_ONE = 1 << 16

# The largest magnitude that can be multiplied by a Q16 constant without leaving
# int64. Any value beyond this raises rather than wrapping.
DOMAIN_MAX = ((1 << 63) - 1 - (Q16_ONE // 2)) // Q16_ONE


class OuterError(Exception):
    pass


def mul_q16(constant: int, values: np.ndarray) -> np.ndarray:
    """`constant`/65536 times `values`, rounded half away from zero.

    Symmetric rounding, like everywhere else integers meet a scale here: an
    arithmetic shift would round toward negative infinity and walk the whole
    model slightly negative over thousands of steps.
    """
    if constant < 0:
        raise OuterError(f"outer: a negative Q16 constant ({constant}) is not a rate")
    peak = int(np.abs(values).max()) if values.size else 0
    if peak > DOMAIN_MAX:
        raise OuterError(
            f"outer: a value of {peak} exceeds the {DOMAIN_MAX} that can be "
            f"multiplied in Q16 without leaving int64")
    sign = np.sign(values)
    magnitude = np.abs(values).astype(np.int64)
    return (sign * ((magnitude * constant + (Q16_ONE // 2)) >> 16)).astype(np.int64)


@dataclass
class OuterOptimizer:
    """Nesterov or classical momentum, in integers, over the whole model.

    The state is one momentum vector and the exponent it is expressed in.
    Carrying the exponent is what lets a round whose contributions came out
    coarser or finer than the last one still accumulate correctly — the momentum
    is realigned by shifting, never by a float round trip.
    """

    momentum_q16: int
    lr_q16: int
    nesterov: bool
    momentum: np.ndarray | None = None
    scale_exp: int = 0
    steps: int = field(default=0)

    def _aligned_momentum(self, size: int, exp: int) -> np.ndarray:
        if self.momentum is None:
            self.momentum = np.zeros(size, dtype=np.int64)
            self.scale_exp = exp
            return self.momentum
        if self.momentum.size != size:
            raise OuterError(
                f"outer: momentum holds {self.momentum.size} values, the update "
                f"has {size}")
        if self.scale_exp == exp:
            return self.momentum
        gap = exp - self.scale_exp
        if gap > 0:
            # The update is coarser than the stored momentum: shift momentum down
            # to meet it, rounding symmetrically.
            from .aggregate import _rounded_shift_down
            self.momentum = _rounded_shift_down(self.momentum, gap)
        else:
            # Finer. Shifting up is exact, but only while it fits.
            peak = int(np.abs(self.momentum).max()) if self.momentum.size else 0
            if peak and peak > DOMAIN_MAX >> (-gap):
                raise OuterError(
                    f"outer: realigning momentum by {-gap} bits would leave the "
                    f"int64 domain")
            self.momentum = (self.momentum << np.int64(-gap)).astype(np.int64)
        self.scale_exp = exp
        return self.momentum

    def step(self, aggregate: np.ndarray, scale_exp: int) -> tuple[np.ndarray, int]:
        """One outer step. Returns the update to apply, at its exponent.

        The update is what gets SUBTRACTED from the weights, matching the
        convention that a contribution is `before - after`: a worker reports how
        far it moved, and the network moves the same way.
        """
        aggregate = np.ascontiguousarray(aggregate, dtype=np.int64)
        momentum = self._aligned_momentum(aggregate.size, scale_exp)

        # m = mu * m + g
        momentum = mul_q16(self.momentum_q16, momentum) + aggregate
        self.momentum = momentum

        if self.nesterov:
            # Look ahead: the update uses where momentum is about to be, which
            # is what makes Nesterov different from classical and is worth the
            # extra multiply.
            direction = aggregate + mul_q16(self.momentum_q16, momentum)
        else:
            direction = momentum

        self.steps += 1
        return mul_q16(self.lr_q16, direction), scale_exp

    def state_bytes(self) -> bytes:
        """The canonical serialization of the optimizer's state.

        Committed to by hash in every checkpoint header, which is what catches
        two producers who applied the same contributions to different momentum.
        The momentum vector itself is never sent — for the mixture it is 235 GB.
        """
        w = (Writer()
             .u32(self.momentum_q16)
             .u32(self.lr_q16)
             .boolean(self.nesterov)
             .u64(self.steps)
             .i64(self.scale_exp)
             .u64(0 if self.momentum is None else self.momentum.size))
        head = w.take()
        if self.momentum is None:
            return head
        return head + self.momentum.astype(">i8").tobytes()

    def state_hash(self) -> bytes:
        return hashlib.sha3_256(self.state_bytes()).digest()


def apply_update(weights: np.ndarray, update: np.ndarray, scale_exp: int
                 ) -> np.ndarray:
    """Weights minus the dequantised update, in float64.

    Dequantising is exact — the scale is a power of two — so the only rounding
    in the whole outer step is the one that puts the result back into bfloat16,
    and it happens once, at the caller.
    """
    if weights.size != update.size:
        raise AggregateError(
            f"outer: {weights.size} weights against {update.size} updates")
    return np.asarray(weights, dtype=np.float64) - np.ldexp(
        np.asarray(update, dtype=np.float64), scale_exp)


def mean_of(values: np.ndarray, count: int) -> np.ndarray:
    """Exposed so a caller averaging outside `aggregate` uses the same rule."""
    return rounded_div(values, count)
