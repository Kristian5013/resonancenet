"""Which arithmetic a round is trained in — pinned, not chosen.

This module exists because of one property the whole protocol rests on: a
verifier re-runs a worker's inner loop and must get BYTE-IDENTICAL output. That
works only if every participant computes identically, and floating-point
arithmetic is not identical across dtypes, kernels, or reduction orders. Flash
attention and the math fallback are each individually reproducible and are NOT
bit-identical to each other — measured, not assumed. A worker free to pick one
would produce an update nobody could reproduce, and would be indistinguishable
from a worker that cheated.

So the numerics are consensus. A round says "bf16 parameters, fp32 accumulation,
flash attention, int8 contributions", every participant either matches it
exactly or sits the round out, and there is no third option where they
participate approximately.

WHAT THIS BUYS, given it looks like a restriction. The alternative — fuzzy
similarity checks on the update, which is what the comparable networks do — is
an open problem by their own account, tuned with empirical thresholds that
honest outliers trip and clever cheats slip under. Pinning the arithmetic is
what makes "prove you did the work" decidable instead of statistical.

THE ESCAPE HATCH IS THE DETERMINISM CLASS. Two workers can verify each other
exactly when their numerics agree, and the class is DERIVED from the canonical
bytes of this object rather than assigned from a registry — so it is
self-certifying, and a network can run several classes at once, each internally
verifiable, without anyone maintaining a table of which is which.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import IntEnum

from ..canon.stream import CanonError, Reader, Writer


class DType(IntEnum):
    """Storage and compute formats a round may pin.

    The values are wire values and must never be reused for something else; a
    round descriptor that meant bf16 last year must not mean fp8 this year.
    """

    BF16 = 1
    FP16 = 2
    FP32 = 3
    FP8_E4M3 = 4
    FP8_E5M2 = 5

    @property
    def bits(self) -> int:
        return {
            DType.BF16: 16, DType.FP16: 16, DType.FP32: 32,
            DType.FP8_E4M3: 8, DType.FP8_E5M2: 8,
        }[self]

    @property
    def torch_name(self) -> str:
        """The torch dtype this maps to, by name.

        By name rather than by object so this module imports without torch — a
        node that only verifies consensus must not need a deep-learning stack,
        and today it does not.
        """
        return {
            DType.BF16: "bfloat16", DType.FP16: "float16", DType.FP32: "float32",
            DType.FP8_E4M3: "float8_e4m3fn", DType.FP8_E5M2: "float8_e5m2",
        }[self]

    @property
    def is_float8(self) -> bool:
        return self in (DType.FP8_E4M3, DType.FP8_E5M2)


class AttentionKernel(IntEnum):
    """Which attention implementation the round is defined in terms of.

    MATH is the portable reduction: reproducible on any device that has the
    dtype at all, and slow. FLASH is the fused kernel: much faster, 16-bit
    inputs only, and it reduces in a different order — which is precisely why
    this is a consensus field and not a performance flag.
    """

    MATH = 1
    FLASH = 2


class ContributionFormat(IntEnum):
    """How a worker's pseudo-gradient is quantised before it goes on the wire.

    All three are integers with a shared power-of-two exponent, because the
    aggregation must be exact integer arithmetic: a sum of int8s with aligned
    exponents is the same number on every node, whereas a sum of floats depends
    on the order the contributions happened to arrive in. The scale is a power
    of two so applying it touches only the exponent of a float and never rounds.

    The value IS the bit width, so the enum reads as what it means at the call
    site and the wire byte stays stable.
    """

    INT4_POW2 = 4
    INT6_POW2 = 6
    INT8_POW2 = 8

    @property
    def max_magnitude(self) -> int:
        """The largest representable magnitude, symmetric about zero.

        Symmetric on purpose: two's complement would give one extra negative
        value, and a quantiser whose range is lopsided biases the mean of the
        update in a direction nobody chose. Proved by
        QuantiseTests.TheRangeIsSymmetricAboutZero.
        """
        return (1 << (self.value - 1)) - 1


@dataclass(frozen=True)
class Numerics:
    """The complete arithmetic contract of a round.

    Frozen because it is hashed: an object whose bytes define a determinism
    class must not be mutable after the class has been derived from it.
    """

    param_dtype: DType
    compute_dtype: DType
    accum_dtype: DType
    attention_kernel: AttentionKernel
    contribution_format: ContributionFormat

    def validate(self) -> None:
        """Internal consistency only — never "does this machine support it".

        Hardware capability is a local question with a local answer, and mixing
        it in here would make an artifact parse on one machine and fail on
        another, which is the one thing a consensus object may never do.
        """
        if self.attention_kernel is AttentionKernel.FLASH and self.compute_dtype.bits != 16:
            # Flash kernels are 16-bit only; asking for fp32 or fp8 gets "No
            # available kernel" at the first forward pass, which is a round
            # nobody can train rather than a round that trains differently.
            raise CanonError(
                f"numerics: flash attention needs a 16-bit compute dtype, "
                f"not {self.compute_dtype.name}"
            )
        if self.accum_dtype.bits < self.compute_dtype.bits:
            # Accumulating narrower than you multiply throws away the precision
            # the multiply just produced, and does it differently depending on
            # how the kernel batches the reduction.
            raise CanonError(
                f"numerics: accumulating in {self.accum_dtype.name} is narrower than "
                f"computing in {self.compute_dtype.name}"
            )
        if self.param_dtype.is_float8 and self.accum_dtype.bits < 32:
            # fp8 parameters carry 3-4 mantissa bits. Without a 32-bit
            # accumulator the update is lost to rounding before it is applied,
            # and the round trains nothing while appearing to work.
            raise CanonError(
                "numerics: fp8 parameters need fp32 accumulation or the update rounds away"
            )

    def serialize(self, w: Writer) -> Writer:
        return (w.u16(self.param_dtype)
                 .u16(self.compute_dtype)
                 .u16(self.accum_dtype)
                 .u16(self.attention_kernel)
                 .u16(self.contribution_format))

    @classmethod
    def parse(cls, r: Reader) -> "Numerics":
        try:
            n = cls(
                param_dtype=DType(r.u16()),
                compute_dtype=DType(r.u16()),
                accum_dtype=DType(r.u16()),
                attention_kernel=AttentionKernel(r.u16()),
                contribution_format=ContributionFormat(r.u16()),
            )
        except ValueError as exc:
            # An unknown enum value is a round from a newer protocol. Refusing is
            # the honest answer: training it would mean guessing what arithmetic
            # somebody else meant.
            raise CanonError(f"numerics: unknown value ({exc})") from exc
        n.validate()
        return n

    def canonical_bytes(self) -> bytes:
        return self.serialize(Writer()).take()

    @property
    def determinism_class(self) -> int:
        """A u32 fingerprint of this arithmetic contract.

        Derived rather than assigned, so two independently configured nodes land
        in the same class if and only if they will actually reproduce each
        other's output, and nobody has to keep a registry honest. Collisions
        would put mutually irreproducible workers in one class, so this reads 32
        bits of SHA3 rather than truncating to 16 as the previous design did.
        Proved by NumericsTests.TheClassChangesWithEveryField.
        """
        return int.from_bytes(
            hashlib.sha3_256(b"rnet/determinism-class/v1" + self.canonical_bytes()).digest()[:4],
            "big",
        )

    def describe(self) -> str:
        return (f"{self.param_dtype.name.lower()} params, "
                f"{self.compute_dtype.name.lower()} compute, "
                f"{self.accum_dtype.name.lower()} accumulate, "
                f"{self.attention_kernel.name.lower()} attention, "
                f"int{self.contribution_format.value} contributions "
                f"(class {self.determinism_class:#010x})")


# The arithmetic round 0 is defined in. bf16 parameters with fp32 accumulation
# is what fits a 400M model at a 16384-token context inside 24 GB, measured on
# the card this was developed against; flash is bit-reproducible within that
# dtype and roughly three times the throughput of the math path at this length.
ROUND0 = Numerics(
    param_dtype=DType.BF16,
    compute_dtype=DType.BF16,
    accum_dtype=DType.FP32,
    attention_kernel=AttentionKernel.FLASH,
    contribution_format=ContributionFormat.INT8_POW2,
)

# The portable class: no fused kernel, so any device with bf16 reproduces it.
# Slower, and the point of it is that a node with an unusual GPU can still
# verify a round rather than being told to trust one.
PORTABLE = Numerics(
    param_dtype=DType.BF16,
    compute_dtype=DType.BF16,
    accum_dtype=DType.FP32,
    attention_kernel=AttentionKernel.MATH,
    contribution_format=ContributionFormat.INT8_POW2,
)
