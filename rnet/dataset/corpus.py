"""Reading the corpus: where a worker gets the text it was told to train on.

The corpus is one file of raw UTF-8. A chunk ends at the first document
separator at or after `target_chunk_bytes` from its start, so BOUNDARIES ARE
DERIVED RATHER THAN STORED — the manifest is three numbers instead of an offset
table for seven million chunks, and any node that scans the file reaches the
same boundaries.

That last clause is the whole risk. Two implementations of one rule is two
chances to get it wrong, and a disagreement about where chunk seven starts is a
disagreement about what everybody trained on, discovered much later. So the
scan lives here, once, and the indexer that builds the Merkle tree calls the
same function.

WHAT A WORKER ACTUALLY ASKS FOR. Not a chunk — a window. The schedule turns a
seed into a chunk index and an offset, and this turns that into tokens. The
worker never chooses any of it, which is what makes the choice checkable.

THE TOKENISATION IS CACHED, THE TEXT IS NOT. An outer step draws hundreds of
windows and revisits chunks; tokenising a megabyte costs about forty
milliseconds, which is nothing once and minutes if repeated. The text itself is
larger than its tokenisation and never wanted twice.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from dataclasses import dataclass, field

from ..dataset.scheduler import chunk_for_window, offset_in_chunk

SEPARATOR = b"\n\n"

# How much to read at a time while looking for a boundary. Large enough that the
# search is one read for a typical chunk, small enough that a pathological file
# with no separator does not pull a gigabyte into memory per step.
SCAN_WINDOW = 1 << 16


class CorpusError(Exception):
    pass


def derive_offsets(path: str, target_chunk_bytes: int, *,
                   progress=None) -> list[int]:
    """Every chunk boundary in the file, by the one rule.

    A chunk runs from its start to the first separator at or after the target;
    the separator belongs to the chunk it ends. Whatever remains after the last
    boundary is the final chunk, however short — which is why the manifest's
    "no chunk shorter than the target" check allows exactly one exception.

    Returns `n_chunks + 1` offsets, the last being the file size.
    """
    size = os.path.getsize(path)
    if size == 0:
        raise CorpusError(f"corpus: {path} is empty")

    offsets = [0]
    with open(path, "rb") as f:
        cursor = 0
        while True:
            search_from = cursor + target_chunk_bytes
            if search_from >= size:
                break
            found = _next_separator(f, search_from, size)
            if found is None or found >= size:
                break
            offsets.append(found)
            cursor = found
            if progress is not None and len(offsets) % 100_000 == 0:
                progress(found, size)
    offsets.append(size)
    return offsets


def _next_separator(f, start: int, size: int) -> int | None:
    """Byte just past the first separator at or after `start`, or None."""
    at = start
    while at < size:
        f.seek(at)
        buf = f.read(min(SCAN_WINDOW, size - at))
        if not buf:
            return None
        hit = buf.find(SEPARATOR)
        if hit != -1:
            return at + hit + len(SEPARATOR)
        if at + len(buf) >= size:
            return None
        # One byte back, so a separator straddling the window edge is seen.
        # Without this a boundary is missed roughly once per 65,536 chunks, and
        # the two implementations disagree about everything after it.
        at += len(buf) - 1
    return None


@dataclass
class LocalCorpus:
    """A corpus held as a file on this machine."""

    path: str
    target_chunk_bytes: int
    tokenizer: object = None
    offsets: list = field(default_factory=list)
    cache_chunks: int = 64
    _tokens: OrderedDict = field(default_factory=OrderedDict)

    @classmethod
    def open(cls, path: str, target_chunk_bytes: int, tokenizer=None, *,
             offsets: list | None = None, progress=None) -> "LocalCorpus":
        """Open it, scanning for boundaries unless they are supplied.

        Scanning seven terabytes takes hours, so a node that has already done it
        passes the offsets back in — from an index file, which is a cache of
        this scan and never a source of truth.
        """
        return cls(path=path, target_chunk_bytes=target_chunk_bytes,
                   tokenizer=tokenizer,
                   offsets=offsets if offsets is not None
                   else derive_offsets(path, target_chunk_bytes,
                                       progress=progress))

    @property
    def n_chunks(self) -> int:
        return max(0, len(self.offsets) - 1)

    @property
    def n_bytes(self) -> int:
        return self.offsets[-1] if self.offsets else 0

    def get(self, index: int) -> bytes:
        """The raw bytes of one chunk."""
        if not 0 <= index < self.n_chunks:
            raise CorpusError(f"corpus: chunk {index} of {self.n_chunks}")
        start, end = self.offsets[index], self.offsets[index + 1]
        with open(self.path, "rb") as f:
            f.seek(start)
            return f.read(end - start)

    def tokens_for_chunk(self, index: int):
        cached = self._tokens.get(index)
        if cached is not None:
            self._tokens.move_to_end(index)
            return cached
        if self.tokenizer is None:
            raise CorpusError("corpus: no tokenizer, so there are no tokens")
        text = self.get(index).decode("utf-8", errors="strict")
        ids = self.tokenizer.encode(text).ids
        if len(self._tokens) >= self.cache_chunks:
            self._tokens.popitem(last=False)
        self._tokens[index] = ids
        return ids

    def window_for_seed(self, seed: bytes, length: int) -> list:
        """The window this seed names: chunk, then offset within it.

        Two draws with separate domains, so knowing one tells a worker nothing
        about the other. A chunk too short to hold the window is an error rather
        than something to clamp — the manifest is meant to make it impossible,
        so arriving here means something upstream is wrong and saying so beats
        quietly training on a truncated window.
        """
        chunk = chunk_for_window(seed, self.n_chunks)
        tokens = self.tokens_for_chunk(chunk)
        start = offset_in_chunk(seed, len(tokens), length - 1)
        return list(tokens[start:start + length])


@dataclass
class RemoteCorpus:
    """A corpus this machine does not hold, fetched through its daemon.

    This is what lets the corpus live on a machine with the disk for it while
    training happens on a machine with the GPU for it. The daemon verifies every
    chunk against `dataset_root` before handing it over, so nothing here has to
    trust where the bytes came from.

    What it does have to handle is waiting: the daemon refuses rather than
    blocks while a fetch is in flight, because it serves every worker from one
    loop. "Not yet" is the mechanism working.
    """

    client: object
    n_chunks: int
    tokenizer: object = None
    cache_chunks: int = 64
    poll_seconds: float = 0.25
    timeout_seconds: float = 120.0
    _tokens: OrderedDict = field(default_factory=OrderedDict)

    def get(self, index: int) -> bytes:
        import time
        deadline = time.monotonic() + self.timeout_seconds
        while True:
            data = self.client.chunk(index)
            if data is not None:
                return data
            if time.monotonic() > deadline:
                raise CorpusError(
                    f"corpus: chunk {index} did not arrive within "
                    f"{self.timeout_seconds:.0f}s — this node has no peer serving "
                    f"the round's corpus")
            time.sleep(self.poll_seconds)

    def tokens_for_chunk(self, index: int):
        cached = self._tokens.get(index)
        if cached is not None:
            self._tokens.move_to_end(index)
            return cached
        if self.tokenizer is None:
            raise CorpusError("corpus: no tokenizer, so there are no tokens")
        ids = self.tokenizer.encode(self.get(index).decode("utf-8")).ids
        if len(self._tokens) >= self.cache_chunks:
            self._tokens.popitem(last=False)
        self._tokens[index] = ids
        return ids

    def window_for_seed(self, seed: bytes, length: int) -> list:
        chunk = chunk_for_window(seed, self.n_chunks)
        tokens = self.tokens_for_chunk(chunk)
        start = offset_in_chunk(seed, len(tokens), length - 1)
        return list(tokens[start:start + length])
