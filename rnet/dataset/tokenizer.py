"""Loading the tokenizer a round pins, and refusing any other.

A tokenizer is not a detail of a worker's setup. Two workers using different
ones produce token ids that mean different strings, so their contributions are
computed over different text while looking identical on the wire — and a
verifier replaying with a third would report a mismatch that means nothing about
either of them.

So the round pins its hash, and this refuses to load a file that does not match.
Not warns: refuses. A worker training under the wrong tokenizer is producing
work nobody can use and cannot be told apart from one that is cheating, which is
the same sentence as everywhere else in this protocol and for the same reason.
"""

from __future__ import annotations

import hashlib
import os

# An all-zero hash means the round has no tokenizer: bytes are tokens. A real
# answer for a byte-level round, not a missing field.
BYTE_LEVEL = bytes(32)


class TokenizerError(Exception):
    pass


class ByteTokenizer:
    """Bytes as tokens, for a round whose tokenizer hash is zero."""

    class Encoding:
        __slots__ = ("ids",)

        def __init__(self, ids):
            self.ids = ids

    def encode(self, text: str):
        return self.Encoding(list(text.encode("utf-8")))

    def decode(self, ids) -> str:
        return bytes(ids).decode("utf-8", errors="replace")


def load(tokenizer_hash: bytes, search: list[str] | None = None):
    """The tokenizer this round names, or an error saying which file was wrong.

    `search` is where to look. The default is the packaged artifact directory,
    because that is where `genesis-emit` puts it and where a fresh clone has it.
    """
    if tokenizer_hash == BYTE_LEVEL:
        return ByteTokenizer()

    if search is None:
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        search = [os.path.join(os.path.dirname(here), "share", "genesis"),
                  os.path.join(here, "share", "genesis")]

    tried = []
    for directory in search:
        path = os.path.join(directory, "tokenizer_round0.json")
        if not os.path.exists(path):
            tried.append(f"{path} (absent)")
            continue
        with open(path, "rb") as f:
            actual = hashlib.sha3_256(f.read()).digest()
        if actual != tokenizer_hash:
            tried.append(f"{path} hashes to {actual.hex()[:16]}…")
            continue
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise TokenizerError(
                "tokenizer: this round pins one and the `tokenizers` package is "
                "not installed — pip install 'resonancenet[train]'") from exc
        return Tokenizer.from_file(path)

    raise TokenizerError(
        f"tokenizer: this round pins {tokenizer_hash.hex()[:16]}… and no file "
        f"matching it was found. Training with a different one produces token "
        f"ids that mean different strings than the network agreed on.\n  "
        + "\n  ".join(tried))
