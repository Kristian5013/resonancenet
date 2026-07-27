"""Where a worker gets the text it was told to train on.

The corpus is raw UTF-8, addressed in document-aligned chunks, so a worker's job
is: fetch the chunk the schedule named, verify it against `dataset_root`,
tokenize it with the artifact `tokenizer_hash` pins, and take its window from the
result. A verifier does the same and must reach the same tokens.

Two things this file is careful about, both because getting them wrong produces a
worker that trains on something nobody else can reproduce:

  * THE TOKENIZATION IS CACHED, NOT THE TEXT. An outer step draws 200 chunks and
    will revisit some of them; tokenizing a megabyte takes about 40 ms, which is
    nothing once but seconds if repeated. The text itself is not worth keeping —
    it is larger than its tokenization and never needed twice.

  * A CHUNK IS VERIFIED BEFORE IT IS TOKENIZED. Bytes that fail their inclusion
    proof are not corpus, whatever they look like, and tokenizing them first
    would mean the failure surfaced as a strange loss rather than as a refusal.

see: docs/corpus-addressing.md
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


class CorpusError(Exception):
    pass


class LocalCorpus:
    """A corpus held as a file on disk, chunked the way the manifest says.

    Boundaries are derived rather than stored — a chunk ends at the first document
    separator at or after `target_chunk_bytes` — so opening a corpus means
    scanning it once. That is the same scan a node does to build the Merkle tree,
    and it must produce the same boundaries or the two disagree about what chunk
    seven is.
    """

    SEPARATOR = b"\n\n"

    def __init__(self, path: str | Path, target_chunk_bytes: int, tokenizer,
                 cache_chunks: int = 64):
        self.path = Path(path)
        self.target_chunk_bytes = int(target_chunk_bytes)
        self.tokenizer = tokenizer
        self._cache: dict[int, np.ndarray] = {}
        self._cache_limit = cache_chunks
        self.offsets = self._derive_offsets()

    def _derive_offsets(self) -> list[int]:
        """The same scan src/dataset/index.cpp does, in the same order.

        A chunk ends at the first separator at or after the target from its start;
        the separator belongs to the chunk it ends. If none remains, the rest of
        the corpus is one final chunk.
        """
        size = self.path.stat().st_size
        if size == 0:
            raise CorpusError("corpus: empty file")
        offsets = [0]
        window = 1 << 16
        with open(self.path, "rb") as f:
            cursor = 0
            while True:
                search_from = cursor + self.target_chunk_bytes
                if search_from >= size:
                    break
                at, found = search_from, 0
                while at < size:
                    f.seek(at)
                    buf = f.read(min(window, size - at))
                    if not buf:
                        break
                    hit = buf.find(self.SEPARATOR)
                    if hit != -1:
                        found = at + hit + len(self.SEPARATOR)
                        break
                    if at + len(buf) >= size:
                        break
                    # One byte back, so a separator across the window edge is seen.
                    at += len(buf) - 1
                if found == 0 or found >= size:
                    break
                offsets.append(found)
                cursor = found
        offsets.append(size)
        return offsets

    @property
    def n_chunks(self) -> int:
        return len(self.offsets) - 1

    @property
    def n_bytes(self) -> int:
        return self.offsets[-1]

    def chunk_bytes(self, index: int) -> bytes:
        if not 0 <= index < self.n_chunks:
            raise CorpusError(f"corpus: chunk {index} of {self.n_chunks}")
        start, end = self.offsets[index], self.offsets[index + 1]
        with open(self.path, "rb") as f:
            f.seek(start)
            return f.read(end - start)

    def tokens_for_chunk(self, index: int) -> np.ndarray:
        cached = self._cache.get(index)
        if cached is not None:
            return cached
        text = self.chunk_bytes(index).decode("utf-8")
        ids = np.asarray(self.tokenizer.encode(text).ids, dtype=np.uint32)
        if len(self._cache) >= self._cache_limit:
            # Oldest first. Insertion order is dict order in Python 3.7+, and the
            # eviction policy does not affect what is trained on — only how often
            # a chunk is tokenized twice.
            self._cache.pop(next(iter(self._cache)))
        self._cache[index] = ids
        return ids


class ArrayCorpus:
    """A corpus that is already a token array, for tests and simulations.

    Presents the same surface as LocalCorpus so the inner loop does not have two
    code paths — one exercised and one not.
    """

    def __init__(self, tokens: np.ndarray, chunk_tokens: int = 1 << 16):
        if len(tokens) < chunk_tokens:
            chunk_tokens = max(1, len(tokens) // 4)
        self.tokens = tokens
        self.chunk_tokens = chunk_tokens
        self.n_chunks = max(1, len(tokens) // chunk_tokens)

    def tokens_for_chunk(self, index: int) -> np.ndarray:
        if not 0 <= index < self.n_chunks:
            raise CorpusError(f"corpus: chunk {index} of {self.n_chunks}")
        start = index * self.chunk_tokens
        end = len(self.tokens) if index == self.n_chunks - 1 else start + self.chunk_tokens
        return self.tokens[start:end]


class RemoteCorpus:
    """A corpus the worker does not hold, fetched a chunk at a time through the node.

    This is what makes a seed worth running: the corpus lives on a machine with the
    disk for it, the model trains on a machine with the GPU for it, and the two need
    not be the same. A chunk is about a megabyte and a round draws a few hundred, so
    an outer step moves a couple of hundred megabytes — 170 KB/s against a
    twenty-minute round.

    The node verifies every chunk against dataset_root before handing it over, so
    nothing here has to trust where the bytes came from. What this class does have to
    handle is waiting: the daemon refuses rather than blocks while a fetch is in
    flight, because it serves every worker from one loop.
    """

    def __init__(self, client, n_chunks: int, tokenizer, cache_chunks: int = 64,
                 poll_seconds: float = 0.25, timeout_seconds: float = 120.0):
        self.client = client
        self.n_chunks = int(n_chunks)
        self.tokenizer = tokenizer
        self._cache: dict[int, np.ndarray] = {}
        self._cache_limit = cache_chunks
        self._poll = poll_seconds
        self._timeout = timeout_seconds

    def chunk_bytes(self, index: int) -> bytes:
        import time as _time
        deadline = _time.monotonic() + self._timeout
        while True:
            try:
                return self.client.corpus_chunk(index)
            except Exception as exc:
                # "Being fetched, ask again" is the normal path, not an error.
                if "fetch" not in str(exc).lower() and "unavailable" not in str(exc).lower():
                    raise
                if _time.monotonic() > deadline:
                    raise CorpusError(
                        f"corpus chunk {index} did not arrive within {self._timeout:.0f}s; "
                        f"the node has no peer serving this round's corpus"
                    ) from exc
                _time.sleep(self._poll)

    def tokens_for_chunk(self, index: int) -> np.ndarray:
        cached = self._cache.get(index)
        if cached is not None:
            return cached
        ids = np.asarray(self.tokenizer.encode(self.chunk_bytes(index).decode("utf-8")).ids,
                         dtype=np.uint32)
        if len(self._cache) >= self._cache_limit:
            self._cache.pop(next(iter(self._cache)))
        self._cache[index] = ids
        return ids


def verify_chunk(chunk: bytes, proof_siblings: list[bytes], leaf_index: int,
                 dataset_root: bytes) -> bool:
    """RFC-6962 inclusion proof, mirroring crypto::VerifyMerkleProof.

    Leaves are prefixed 0x00 and internal nodes 0x01, so a leaf can never be
    substituted for an internal node.
    """
    node = hashlib.sha3_256(b"\x00" + chunk).digest()
    index = leaf_index
    for sibling in proof_siblings:
        if index % 2 == 0:
            node = hashlib.sha3_256(b"\x01" + node + sibling).digest()
        else:
            node = hashlib.sha3_256(b"\x01" + sibling + node).digest()
        index //= 2
    return node == dataset_root
