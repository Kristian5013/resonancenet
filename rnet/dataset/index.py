"""Indexing a corpus: from a file of text to a root a round can pin.

Find the boundaries by the rule in `corpus.py` — the same function, called
rather than reimplemented, because two implementations of one rule is two
chances to disagree about what chunk seven is — hash each chunk into a Merkle
leaf, and fold the leaves into the root.

It is not a streaming pass, and the difference matters for what it costs.
`derive_offsets` seeks to `target_chunk_bytes` past each boundary and reads a
64 KiB window looking for the separator, so finding the boundaries touches a few
percent of the file rather than all of it. Hashing then reads every byte once.

WHY THE OFFSETS ARE CACHED AND NEVER TRUSTED. Scanning seven terabytes is hours
of I/O, so the index file exists to avoid repeating it. But it is a cache of a
derivation, not a source of truth: it records the file size and the target it
was built at, and a node that loads it against a different file rebuilds rather
than believing it. An index that could disagree with the corpus and win would be
a way to make two nodes train on different text while agreeing on a root.

THE ARITHMETIC IS NOT THE BOTTLENECK, and it is worth saying because the
opposite is assumed. Measured on the machine this was written for:

    boundary scan (bytes.find over a window) 5,185 MB/s
    SHA3 across 24 threads (hashlib)        13,927 MB/s
    the disk the corpus is on                  384 MB/s

Fourteen and thirty-six times the disk. The whole 7.36 TB is about nine minutes
of computation and hours of reading.
"""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from ..consensus.objects import DatasetManifest
from ..crypto import merkle
from .corpus import derive_offsets

INDEX_MAGIC = "rnet-corpus-index/v1"

# Chunks hashed per batch. Enough to keep every thread fed, small enough that
# the batch itself is a bounded read rather than the whole corpus.
BATCH = 256


class IndexError_(Exception):
    pass


@dataclass
class CorpusIndex:
    path: str
    target_chunk_bytes: int
    n_bytes: int
    offsets: list
    root: bytes

    @property
    def n_chunks(self) -> int:
        return max(0, len(self.offsets) - 1)

    def manifest(self, tokenizer_hash: bytes) -> DatasetManifest:
        return DatasetManifest(
            dataset_root=self.root, tokenizer_hash=tokenizer_hash,
            n_bytes=self.n_bytes, target_chunk_bytes=self.target_chunk_bytes,
            n_chunks=self.n_chunks)

    def save(self, path: str) -> None:
        """Atomically. A torn index would claim boundaries the corpus lacks."""
        payload = {
            "magic": INDEX_MAGIC,
            "corpus": os.path.basename(self.path),
            "n_bytes": self.n_bytes,
            "target_chunk_bytes": self.target_chunk_bytes,
            "root": self.root.hex(),
            "offsets": self.offsets,
        }
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str, corpus_path: str,
             target_chunk_bytes: int) -> "CorpusIndex | None":
        """Read a cached index, or None if it does not match this corpus.

        None rather than an exception: a stale index is an ordinary thing to
        find, and the answer is to rebuild rather than to stop.
        """
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, ValueError):
            return None
        if data.get("magic") != INDEX_MAGIC:
            return None
        if data.get("target_chunk_bytes") != target_chunk_bytes:
            return None
        try:
            if data["n_bytes"] != os.path.getsize(corpus_path):
                return None
        except OSError:
            return None
        return cls(path=corpus_path, target_chunk_bytes=target_chunk_bytes,
                   n_bytes=int(data["n_bytes"]), offsets=list(data["offsets"]),
                   root=bytes.fromhex(data["root"]))


def _leaf(args) -> bytes:
    path, start, end = args
    with open(path, "rb") as f:
        f.seek(start)
        return merkle.leaf_hash(f.read(end - start))


def build_index(corpus_path: str, target_chunk_bytes: int, *,
                offsets: list | None = None, workers: int = 0,
                progress=None) -> CorpusIndex:
    """Scan, hash, and fold. Returns the index and its root."""
    if offsets is None:
        offsets = derive_offsets(corpus_path, target_chunk_bytes,
                                 progress=progress)
    n_chunks = len(offsets) - 1
    if n_chunks <= 0:
        raise IndexError_(f"index: {corpus_path} yielded no chunks")

    if workers == 0:
        workers = max(1, min(24, (os.cpu_count() or 2)))

    leaves: list = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for at in range(0, n_chunks, BATCH):
            batch = [(corpus_path, offsets[i], offsets[i + 1])
                     for i in range(at, min(at + BATCH, n_chunks))]
            leaves.extend(pool.map(_leaf, batch))
            if progress is not None:
                progress(len(leaves), n_chunks)

    return CorpusIndex(path=corpus_path, target_chunk_bytes=target_chunk_bytes,
                       n_bytes=offsets[-1], offsets=offsets,
                       root=merkle.root(leaves))


def proof_for(index: CorpusIndex, chunk: int) -> merkle.Proof:
    """An inclusion proof for one chunk, rebuilt from the file.

    The leaves are not kept in memory — for seven million chunks that is 223 MB
    of hashes, which is affordable but pointless when serving one proof means
    reading the corpus anyway.
    """
    n = index.n_chunks
    if not 0 <= chunk < n:
        raise IndexError_(f"index: chunk {chunk} of {n}")
    leaves = [_leaf((index.path, index.offsets[i], index.offsets[i + 1]))
              for i in range(n)]
    return merkle.build_proof(leaves, chunk)


def verify_chunk(data: bytes, proof: merkle.Proof, root: bytes) -> bool:
    return merkle.verify_proof(merkle.leaf_hash(data), proof, root)
