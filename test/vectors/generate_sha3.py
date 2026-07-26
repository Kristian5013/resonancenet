#!/usr/bin/env python3
"""Generates the SHA3-256 vector file from an independent implementation.

Every hash in this protocol is SHA3-256: the genesis anchor, Merkle roots over the
corpus, contribution payloads, checkpoint weights, producer election, slashing
evidence. One primitive carries all of it, and it is a vendored 191-line file.

Before this existed it was verified with three vectors. Three. If that
implementation were wrong at some length boundary, nothing here would have
noticed — until the day another node's implementation disagreed and the network
split with no symptom to point at.

The vectors come from CPython's `hashlib`, which wraps the Keccak reference code.
That makes this a genuine cross-implementation check rather than a restatement of
whatever our vendored copy happens to produce; a test generated from the code
under test proves only that it is consistent with itself.

Three kinds of vector, each catching something the others cannot:

  * EVERY LENGTH from 0 to 600. The SHA3-256 rate is 136 bytes, so this covers
    four full absorption blocks and every boundary within them. Padding and
    absorption bugs live at exactly these lengths and nowhere else, and picking
    which boundary to test by hand is guessing.

  * EXPLICIT RANDOM MESSAGES at the interesting lengths. A patterned input can
    mask a byte-order error by being too regular — reverse a word of a repeating
    pattern and it may still hash the same way.

  * THE MONTE CARLO PROCEDURE. A hundred thousand chained hashes, each one's
    output the next one's input. Single-shot vectors cannot catch state that is
    not cleared between calls, or a permutation wrong in a way that happens to
    cancel on the first block. Here an error at any point changes everything
    after it, so one mismatch at the end proves the whole chain.

Usage:
    python3 test/vectors/generate_sha3.py > test/vectors/sha3_256.txt
"""

from __future__ import annotations

import hashlib
import random
import sys

# Covers four full 136-byte rates. Wide enough that no boundary has to be chosen.
MAX_PATTERN_LENGTH = 600

# Lengths where an explicit random message is worth the file size: the rate
# boundaries, the word boundaries, and a few sizes past them.
EXPLICIT_LENGTHS = [1, 2, 3, 63, 64, 65, 127, 135, 136, 137, 271, 272, 273, 500, 1000, 4096]

# The NIST CAVP Monte Carlo shape: 100 outer rounds of 1000 chained hashes.
MONTE_CARLO_OUTER = 100
MONTE_CARLO_INNER = 1000
MONTE_CARLO_CHECKPOINTS = (0, 9, 49, 99)

# Fixed so the file regenerates identically. A generator that produced different
# vectors each run would make a diff meaningless.
EXPLICIT_SEED = 20260726

# The SHA3-256 of the empty string. Used as the Monte Carlo seed because it is a
# published constant anyone can check rather than a number chosen here.
MONTE_CARLO_SEED = "a7ffc6f8bf1ed76651c14756a061d662f580ff4de43b49fa82d80a4b80f8434a"


def patterned(length: int) -> bytes:
    """A deterministic message, so the file stores only lengths and digests."""
    return bytes((i * 31 + 7) % 256 for i in range(length))


def main() -> int:
    out = sys.stdout
    out.write(
        "# SHA3-256 vectors generated from CPython hashlib, which wraps the Keccak\n"
        "# reference implementation — a cross-implementation check, not a restatement\n"
        "# of whatever the vendored code produces.\n"
        "#\n"
        "# Regenerate: python3 test/vectors/generate_sha3.py > test/vectors/sha3_256.txt\n"
        "#\n"
        "#   P <len> <digest>       message byte i = (i*31+7) mod 256\n"
        "#   E <msg-hex> <digest>   explicit message\n"
        "#   C <round> <digest>     Monte Carlo state after (round+1)*1000 chained hashes\n"
    )

    for length in range(MAX_PATTERN_LENGTH + 1):
        out.write(f"P {length} {hashlib.sha3_256(patterned(length)).hexdigest()}\n")

    rng = random.Random(EXPLICIT_SEED)
    for length in EXPLICIT_LENGTHS:
        message = bytes(rng.randrange(256) for _ in range(length))
        out.write(f"E {message.hex()} {hashlib.sha3_256(message).hexdigest()}\n")

    digest = bytes.fromhex(MONTE_CARLO_SEED)
    for outer in range(MONTE_CARLO_OUTER):
        for _ in range(MONTE_CARLO_INNER):
            digest = hashlib.sha3_256(digest).digest()
        if outer in MONTE_CARLO_CHECKPOINTS:
            out.write(f"C {outer} {digest.hex()}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
