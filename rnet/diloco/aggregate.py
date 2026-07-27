"""Adding contributions up, in integers, so everyone gets the same sum.

This is the operation the outer loop is built around, and it has exactly one
requirement that is not obvious: THE RESULT MUST NOT DEPEND ON THE ORDER THE
CONTRIBUTIONS ARRIVED IN. Floating-point addition is not associative, so a sum
of dequantised floats gives a different answer on a node that received Bob
before Alice — and every node would be correct about its own arithmetic while
disagreeing about the checkpoint. That is a chain split arriving through a
rounding rule.

Integers are associative. So contributions are summed as integers, and the only
step that needs care is what to do when two workers chose different exponents:
they are ALIGNED BY SHIFTING, never by rescaling through a float. Shifting the
finer contribution up to the coarser one's scale is exact; converting both to
float and back is not.

WHY A DIVERGENT EXPONENT IS REFUSED RATHER THAN ALIGNED. Aligning a contribution
whose scale is 2^40 away means shifting it left by forty bits — the values would
dominate the sum by a trillion to one, and a worker could do it deliberately.
Past a bound, a scale that far from its peers is either a broken run or an
attack, and both are better refused than averaged in.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..consensus.numerics import ContributionFormat

# The widest exponent gap two honest contributions can show. Twenty-four bits is
# eight orders of magnitude between the largest updates two workers produced in
# the same round from the same weights; wider than that is not disagreement
# about scale, it is a different quantity.
MAX_ALIGN_SHIFT = 24

# Room for the sum itself. int64 holds about 9.2e18; the aligned values are at
# most max_magnitude << MAX_ALIGN_SHIFT each, so this many contributions can be
# added without approaching the edge, and more are refused rather than wrapped.
MAX_CONTRIBUTORS = 1 << 20


class AggregateError(Exception):
    pass


@dataclass(frozen=True)
class Contribution:
    """One worker's quantised update, with the exponent it chose."""

    values: np.ndarray
    scale_exp: int
    fmt: ContributionFormat
    worker_id: int = 0


def sum_contributions(contributions: list[Contribution]) -> tuple[np.ndarray, int]:
    """The integer sum, at the coarsest exponent present.

    Returns the summed integers and the exponent they are in, so the caller can
    dequantise once at the end rather than per contribution.
    """
    if not contributions:
        raise AggregateError("aggregate: nothing to aggregate")
    if len(contributions) > MAX_CONTRIBUTORS:
        raise AggregateError(
            f"aggregate: {len(contributions)} contributions exceeds "
            f"{MAX_CONTRIBUTORS}")

    size = contributions[0].values.size
    for c in contributions:
        if c.values.size != size:
            raise AggregateError(
                f"aggregate: worker {c.worker_id} sent {c.values.size} values, "
                f"the first sent {size}")

    # The coarsest exponent wins, because shifting a finer contribution UP is
    # exact while shifting a coarser one DOWN would discard its low bits.
    target = max(c.scale_exp for c in contributions)
    for c in contributions:
        gap = target - c.scale_exp
        if gap > MAX_ALIGN_SHIFT:
            raise AggregateError(
                f"aggregate: worker {c.worker_id} is at exponent {c.scale_exp} "
                f"against {target} — {gap} bits apart, past the {MAX_ALIGN_SHIFT} "
                f"allowed. That is a different quantity, not a different scale.")

    total = np.zeros(size, dtype=np.int64)
    for c in contributions:
        gap = target - c.scale_exp
        # Right-shift toward the coarser scale, rounding symmetrically so the
        # aggregate is not biased toward zero by the alignment itself.
        if gap:
            total += _rounded_shift_down(c.values, gap)
        else:
            total += c.values
    return total, target


def _rounded_shift_down(values: np.ndarray, bits: int) -> np.ndarray:
    """values / 2**bits, rounded half away from zero, in integers.

    Symmetric about zero on purpose: an arithmetic right shift rounds toward
    negative infinity, which would drag every aggregate slightly negative — a
    bias no one chose and nobody would find by looking at the loss.
    """
    half = np.int64(1) << np.int64(bits - 1)
    sign = np.sign(values)
    return (sign * ((np.abs(values) + half) >> np.int64(bits))).astype(np.int64)


def average_contributions(contributions: list[Contribution]
                          ) -> tuple[np.ndarray, int]:
    """The mean, as integers. What the outer step actually consumes.

    A sum, not a race: every contribution is in the result, and no worker's work
    is dropped for arriving second.
    """
    total, exp = sum_contributions(contributions)
    return rounded_div(total, len(contributions)), exp


def rounded_div(values: np.ndarray, divisor: int) -> np.ndarray:
    """Integer division rounding half away from zero, symmetric about zero."""
    if divisor <= 0:
        raise AggregateError(f"aggregate: divisor {divisor} must be positive")
    sign = np.sign(values)
    return (sign * ((np.abs(values) + divisor // 2) // divisor)).astype(np.int64)


def contribution_root(ids: list[bytes]) -> bytes:
    """The Merkle root over the contribution ids that went into a checkpoint.

    Sorted, so two producers that received the same contributions in different
    orders commit to the same root. Without that, "every contribution was
    aggregated" would be a claim about arrival order rather than about content.
    """
    from ..crypto import merkle
    if len(set(ids)) != len(ids):
        raise AggregateError("aggregate: the same contribution is cited twice")
    return merkle.root([merkle.leaf_hash(i) for i in sorted(ids)])
