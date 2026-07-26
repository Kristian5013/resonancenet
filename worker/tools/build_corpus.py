#!/usr/bin/env python3
"""Builds the tokenized corpus that a round's `dataset_root` commits to.

The output of this script becomes a consensus value. Once its Merkle root is
pinned in a genesis artifact, every worker in the network derives training
windows into these exact bytes, and every verifier replays another worker's batch
from them. So the requirement is not "produce a reasonable corpus" — it is
"produce the same bytes as anyone else running this on the same inputs".

Everything below that could differ between two runs is therefore fixed:

  * PINNED REVISIONS. A HuggingFace dataset is a moving branch: files get
    re-uploaded, fixed and added. `--revision` takes a commit SHA, never `main`.
    Without it, someone rebuilding this corpus in a year gets different bytes and
    can neither reproduce the root nor verify a single chunk.

  * DETERMINISTIC ORDER. Shards are processed in lexicographic filename order and
    rows in the order they appear in the file. No parallelism reorders anything —
    tokenization is parallel within a batch, but batches are written in sequence.

  * ONE SEPARATOR RULE. Token 0 (`<|eos|>`) is appended after every document and
    nowhere else. Whether documents are separated at all is invisible in the
    token count and changes every byte after the first document.

  * LITTLE-ENDIAN uint32. This is the one place in the project that is not
    big-endian, because the C++ side reads the token file directly rather than
    through the canonical container (see ReadWindow in src/dataset/manifest.cpp:
    `b[0] | b[1]<<8 | b[2]<<16 | b[3]<<24`). Writing big-endian here would produce
    a corpus that hashes fine and trains on nonsense.

  * THE TOKENIZER IS VERIFIED, NOT ASSUMED. Its hash is checked against the value
    pinned in the round descriptor before a single document is encoded. A corpus
    built with a different tokenizer is a different corpus wearing the right name.

Disk discipline: shards are deleted as they are consumed, so peak usage is the
output file plus one shard rather than the whole raw dataset. That is what makes
a multi-terabyte corpus buildable on a disk sized for the result.

Resumable, because a run measured in days will be interrupted. State is written
after every shard; re-running continues from the last completed one.

Usage:
    build_corpus.py --config corpus.json --out data/corpus.bin
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_files
from tokenizers import Tokenizer

# Token 0 in a BPE vocabulary. Appended after every document and nowhere else.
EOS_TOKEN = 0

# What separates documents in a BYTE-LEVEL corpus.
#
# A byte vocabulary has no spare id: all 256 values are real bytes of real text,
# so there is nothing to reserve as a separator. A blank line is the separator the
# text itself already uses, it costs two bytes, and it needs no agreement beyond
# this constant — a worker reading the corpus sees exactly what was written.
BYTE_SEPARATOR = b"\n\n"

# Documents encoded per call. Large enough that the Rust tokenizer's parallelism
# pays off, small enough that a batch of long documents does not spike memory.
BATCH_DOCUMENTS = 1000

# Downloads in flight at once.
#
# One at a time was enough on a c7i.2xlarge, where eight cores took about five
# minutes to tokenize a shard that downloaded in eight seconds. It is not enough
# on a machine with sixty-four cores and a hundred-gigabit link: tokenizing gets
# eight times faster while a single HTTP stream does not, so the cores go back to
# waiting. Sixteen streams is what saturates the link without the ordering below
# becoming the bottleneck instead.
DOWNLOAD_WORKERS = 8

# Shards allowed on disk ahead of the one being tokenized. Bounds peak disk to
# this many shards — about a gigabyte each for FineWeb-Edu — on top of the output.
PREFETCH_SHARDS = 24

# How many times a shard is retried before the run gives up on it.
#
# High, and deliberately so. A shard CANNOT be skipped: its bytes and its position
# are both part of dataset_root, so a corpus missing one is a different corpus
# that looks exactly like the right one. The only alternatives to retrying are
# stopping — which threw away five hours of a run at shard 1595 of 2410, on an
# internal error inside the Hub's transfer client rather than on anything wrong
# with the file — or corrupting the output. So it retries for about half an hour
# before admitting defeat, and admits it by stopping cleanly with the state
# written, not by unwinding out of the middle of a write.
DOWNLOAD_ATTEMPTS = 12
MAX_BACKOFF_SECONDS = 300


@dataclass
class SourceSpec:
    """One dataset to draw from, pinned to an exact revision."""

    repo: str
    revision: str          # a commit SHA — never a branch name
    text_column: str       # column holding the document text
    include: list[str]     # path prefixes to take, e.g. ["data/CC-MAIN-2025-26/"]
    max_tokens: int        # stop after this many, so the mix is what was intended
    # Stack v3 rows are whole repositories with their files in an array rather
    # than one document per row; this names that array when it applies.
    files_column: str | None = None
    files_text_key: str | None = None


@dataclass
class BuildState:
    """Everything needed to continue an interrupted run."""

    tokens_written: int = 0
    # How many bytes of the output file this state accounts for.
    #
    # Without it, "resume" means "append after whatever is there", and whatever is
    # there may include a half-written shard that no state records — so the shard
    # is tokenized again and the corpus contains a partial duplicate. That is
    # invisible in every number the build prints, and it changes dataset_root.
    #   proven by: test_corpus_builder.py test_resume_after_a_crash_mid_shard
    bytes_written: int = 0
    source_index: int = 0
    completed_shards: list[str] = None
    source_tokens: dict = None
    raw_bytes_read: int = 0

    def __post_init__(self):
        if self.completed_shards is None:
            self.completed_shards = []
        if self.source_tokens is None:
            self.source_tokens = {}


def load_tokenizer(path: Path, expected_hash: str | None) -> Tokenizer:
    """Loads the pinned tokenizer, or returns None for a byte-level corpus.

    `None` is not a fallback: it is the byte-level mode, which the round descriptor
    spells the same way — an all-zero tokenizer_hash means there is no artifact
    because the corpus is the text.
    """
    """Loads the tokenizer and refuses one that is not the pinned one.

    A corpus built with a different tokenizer is a different corpus wearing the
    right name: the token ids mean other strings, and nothing downstream can tell.
    """
    if path is None:
        print("byte-level corpus: no tokenizer, the file IS the text")
        return None
    raw = path.read_bytes()
    actual = hashlib.sha3_256(raw).hexdigest()
    if expected_hash and actual != expected_hash:
        raise SystemExit(
            f"tokenizer mismatch:\n"
            f"  file:   {actual}\n"
            f"  pinned: {expected_hash}\n"
            f"Building with this would produce a corpus whose token ids mean "
            f"different strings than the network agreed on."
        )
    model = json.loads(raw)["model"]
    vocab = model.get("vocab_size") or len(model["vocab"])
    print(f"tokenizer {actual[:16]}… vocab {vocab}")
    return Tokenizer.from_file(str(path))


def shard_list(source: SourceSpec) -> list[str]:
    """The shards to read, in lexicographic order.

    Order is part of the output: the same shards taken in a different sequence
    produce a different file and therefore a different root.
    """
    files = list_repo_files(source.repo, repo_type="dataset", revision=source.revision)
    chosen = [
        f
        for f in files
        if f.endswith(".parquet") and any(f.startswith(p) for p in source.include)
    ]
    return sorted(chosen)


class ShardFetcher:
    """Downloads shards ahead of the loop that consumes them, in parallel.

    Downloading and tokenizing are both slow and use different resources — one the
    network, the other every core — so doing them in sequence leaves each idle
    while the other works.

    ORDER IS PART OF THE OUTPUT, so although downloads finish in whatever order the
    network delivers them, they are handed to the consumer strictly in shard-list
    order: futures go into a deque when submitted and are read from its front. A
    shard that arrives early waits its turn.
      proven by: test_corpus_builder.py test_parallel_downloads_preserve_order
    """

    def __init__(self, source: SourceSpec, shards: list[str], cache_dir: Path, skip: set[str]):
        self.source = source
        self.cache_dir = cache_dir
        self.pending = [s for s in shards if s not in skip]

    def _download(self, shard: str) -> str:
        last: Exception | None = None
        for attempt in range(DOWNLOAD_ATTEMPTS):
            try:
                return hf_hub_download(
                    self.source.repo, shard, repo_type="dataset",
                    revision=self.source.revision, cache_dir=str(self.cache_dir),
                )
            except Exception as exc:            # noqa: BLE001 - retried, then reported
                last = exc
                if attempt + 1 < DOWNLOAD_ATTEMPTS:
                    delay = min(2 ** attempt, MAX_BACKOFF_SECONDS)
                    print(f"  {shard}: {type(exc).__name__}, retrying in {delay}s "
                          f"({attempt + 1}/{DOWNLOAD_ATTEMPTS})", file=sys.stderr)
                    time.sleep(delay)
        raise RuntimeError(f"{shard}: failed after {DOWNLOAD_ATTEMPTS} attempts") from last

    def __iter__(self):
        with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
            remaining = iter(self.pending)
            inflight: collections.deque = collections.deque()

            def fill():
                while len(inflight) < PREFETCH_SHARDS:
                    shard = next(remaining, None)
                    if shard is None:
                        return
                    inflight.append((shard, pool.submit(self._download, shard)))

            fill()
            while inflight:
                shard, future = inflight.popleft()
                local = future.result()     # re-raises a permanent failure here
                fill()
                yield shard, local


def documents_in(table, source: SourceSpec):
    """Yields document texts from one parquet table, in row order.

    Two shapes are handled. Most corpora put one document per row. Stack v3 puts a
    whole repository per row with its files in an array, so each file becomes a
    document — repository-level grouping is deliberate there and preserved here.
    """
    if source.files_column:
        for repo_files in table.column(source.files_column).to_pylist():
            if not repo_files:
                continue
            for entry in repo_files:
                text = entry.get(source.files_text_key) if isinstance(entry, dict) else None
                if text:
                    yield text
    else:
        for text in table.column(source.text_column).to_pylist():
            if text:
                yield text


def append_text(out, documents: list[str]) -> tuple[int, int]:
    """Writes documents as UTF-8, separated by a blank line. Returns (bytes, bytes).

    Both halves of the pair are the same number here, and deliberately so: in a
    byte-level corpus a token IS a byte, so the token count and the byte count
    cannot disagree. Nothing is encoded, counted twice, or converted.
    """
    blob = BYTE_SEPARATOR.join(d.encode("utf-8") for d in documents) + BYTE_SEPARATOR
    out.write(blob)
    return len(blob), len(blob)


def append_tokens(out, tokenizer: Tokenizer, documents: list[str]) -> tuple[int, int]:
    """Encodes a batch and appends it. Returns (tokens written, raw bytes read).

    `encode_batch` is parallel inside the Rust tokenizer, but the results are
    written in the order the documents were given — the parallelism is over the
    work, not over the output.
    """
    encodings = tokenizer.encode_batch(documents)
    raw_bytes = sum(len(d.encode("utf-8")) for d in documents)

    total = sum(len(e.ids) + 1 for e in encodings)   # +1 for the separator
    buffer = np.empty(total, dtype="<u4")            # little-endian; see the module docstring
    cursor = 0
    for encoding in encodings:
        n = len(encoding.ids)
        buffer[cursor : cursor + n] = np.asarray(encoding.ids, dtype="<u4")
        buffer[cursor + n] = EOS_TOKEN
        cursor += n + 1

    out.write(buffer.tobytes())
    return total, raw_bytes


# Parallel chunked transfer. Measured against the default on the same host:
# 45 MB/s single-stream versus 194 MB/s — the difference between fifty hours and
# thirteen for a nine-terabyte corpus.
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")


def write_state(path: Path, state: "BuildState") -> None:
    """Replaces the state file atomically.

    write_text truncates and then writes, so a process dying in between leaves an
    empty or half-written file — and the resume that follows cannot parse it, which
    turns a recoverable interruption into a corpus rebuilt from zero.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(asdict(state)))
    os.replace(tmp, path)


def free_bytes(path: Path) -> int:
    stat = os.statvfs(path)
    return stat.f_bavail * stat.f_frsize


def build(config_path: Path, out_path: Path, cache_dir: Path, keep_raw: bool,
          list_shards=shard_list, fetcher_factory=None) -> None:
    """Builds the token file.

    `list_shards` and `fetcher_factory` exist so the tokenizing half can be driven
    from local files. Without a seam here the only way to exercise the code that
    produces a consensus value is to download a terabyte first, which means it was
    never exercised at all.
    """
    if fetcher_factory is None:
        fetcher_factory = ShardFetcher
    config = json.loads(config_path.read_text())
    sources = [SourceSpec(**s) for s in config["sources"]]

    tokenizer_path = config.get("tokenizer")
    tokenizer = load_tokenizer(Path(tokenizer_path) if tokenizer_path else None,
                               config.get("tokenizer_hash"))

    state_path = out_path.with_suffix(".state.json")
    state = BuildState(**json.loads(state_path.read_text())) if state_path.exists() else BuildState()
    if state.tokens_written:
        print(f"resuming: {state.tokens_written:,} tokens already written, "
              f"{len(state.completed_shards)} shards done")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Append, so a resumed run continues the same file rather than starting a
    # second one. Truncating here would silently discard days of work.
    mode = "ab" if state.tokens_written else "wb"

    # Cut back to the last byte the state accounts for. Anything past it belongs
    # to a shard that was interrupted partway and is about to be redone, so
    # keeping it would duplicate a fragment of it.
    if state.bytes_written and out_path.exists():
        actual = out_path.stat().st_size
        if actual > state.bytes_written:
            print(f"discarding {actual - state.bytes_written:,} unaccounted bytes from an "
                  f"interrupted shard")
            os.truncate(out_path, state.bytes_written)
        elif actual < state.bytes_written:
            raise SystemExit(
                f"the output file is {actual:,} bytes but the state records "
                f"{state.bytes_written:,}. Something truncated it. Refusing to append to a "
                f"corpus that is already shorter than its own record."
            )
    started = time.time()

    with open(out_path, mode) as out:
        for source_index, source in enumerate(sources):
            if source_index < state.source_index:
                continue
            state.source_index = source_index
            written_here = state.source_tokens.get(source.repo, 0)

            shards = list_shards(source)
            print(f"\n{source.repo} @ {source.revision[:12]}: {len(shards)} shards, "
                  f"target {source.max_tokens:,} tokens")

            done = set(state.completed_shards)
            fetcher = fetcher_factory(source, shards, cache_dir, done)

            for shard, local in fetcher:
                if written_here >= source.max_tokens:
                    print(f"  reached {written_here:,} tokens; moving on")
                    break

                table = pq.read_table(local)

                batch: list[str] = []
                shard_tokens = 0
                for text in documents_in(table, source):
                    batch.append(text)
                    if len(batch) >= BATCH_DOCUMENTS:
                        n, raw = (append_text(out, batch) if tokenizer is None
                                  else append_tokens(out, tokenizer, batch))
                        shard_tokens += n
                        state.raw_bytes_read += raw
                        batch.clear()
                if batch:
                    n, raw = (append_text(out, batch) if tokenizer is None
                              else append_tokens(out, tokenizer, batch))
                    shard_tokens += n
                    state.raw_bytes_read += raw

                written_here += shard_tokens
                state.tokens_written += shard_tokens
                state.source_tokens[source.repo] = written_here
                state.completed_shards.append(shard)

                # Deleted as consumed, so peak disk is the output plus the
                # prefetch window rather than the whole raw dataset.
                #
                # BOTH the symlink and the blob it points at. hf_hub_download
                # returns a path under snapshots/ that is a symlink into blobs/,
                # where the bytes actually are, so removing what it handed back
                # frees nothing at all. This filled a 3.4 TB disk at shard 1595 of
                # 2410 and surfaced as "Internal Writer Error: receiver dropped"
                # from the transfer client — five hours lost to a message that
                # named neither the disk nor the space.
                #   proven by: test_corpus_builder.py
                #   test_consumed_shards_free_their_disk_space
                if not keep_raw:
                    for path in {local, os.path.realpath(local)}:
                        try:
                            os.remove(path)
                        except OSError:
                            pass

                # Flushed and on the platter BEFORE the state claims it. The other
                # order records bytes that a power loss would take away, and the
                # resume would then trust a length the file does not have.
                out.flush()
                os.fsync(out.fileno())
                state.bytes_written = out.tell()
                write_state(state_path, state)

                # Stopped cleanly rather than dying mid-write. A corpus is worth
                # days of compute and the state file is only consistent between
                # shards; running the volume to zero would leave a truncated file
                # with no record of where it stopped.
                remaining = free_bytes(out_path.parent)
                if remaining < 20 * 1024 ** 3:
                    print(f"\nstopping: {remaining / 1e9:.1f} GB free, below the 20 GB reserve.")
                    print(f"{state.tokens_written:,} tokens written and recorded; "
                          f"grow the volume and re-run to continue.")
                    return

                elapsed = time.time() - started
                ratio = state.raw_bytes_read / max(state.tokens_written, 1)
                print(f"  {shard.split('/')[-1]}: +{shard_tokens:,} → {state.tokens_written:,} "
                      f"({ratio:.2f} raw bytes/token, {state.tokens_written / max(elapsed, 1):,.0f} tok/s)")

    ratio = state.raw_bytes_read / max(state.tokens_written, 1)
    size = out_path.stat().st_size
    # The file is four bytes per token or it is not the file the count describes.
    # Cheap, and the only check that catches a duplicated or truncated fragment
    # before its root is pinned.
    width = 1 if tokenizer is None else 4
    if size != state.tokens_written * width:
        raise SystemExit(
            f"the corpus is {size:,} bytes but claims {state.tokens_written:,} tokens "
            f"({state.tokens_written * width:,} bytes). Do not pin a root from this file."
        )
    print(f"\n{state.tokens_written:,} tokens, {size / 1e9:.1f} GB on disk")
    print(f"{ratio:.3f} raw bytes per token with THIS tokenizer")
    # The number that sizes a disk. Published token counts use other tokenizers,
    # and a larger vocabulary yields fewer tokens for the same text — so a plan
    # built on someone else's count is a plan built on someone else's vocabulary.
    print(f"→ {ratio * 1e12 / 1e12:.2f} TB of raw text per trillion tokens, "
          f"{4e12 / 1e12:.0f} TB on disk per trillion tokens")

    meta = {
        "n_tokens": state.tokens_written,
        "dtype": "uint8" if tokenizer is None else "uint32",
        "byte_order": "little-endian",
        "eos": None if tokenizer is None else EOS_TOKEN,
        "separator": BYTE_SEPARATOR.decode() if tokenizer is None else None,
        "tokenizer_hash": config.get("tokenizer_hash"),
        "raw_bytes_per_token": round(ratio, 4),
        "sources": [
            {"repo": s.repo, "revision": s.revision, "tokens": state.source_tokens.get(s.repo, 0)}
            for s in sources
        ],
    }
    out_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nwrote {out_path.with_suffix('.meta.json')}")
    print(f"next:  rnet-tool dataset-build --file {out_path} --out {out_path.with_suffix('')}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path("/tmp/rnet-corpus-cache"))
    parser.add_argument("--keep-raw", action="store_true",
                        help="do not delete shards after reading them")
    args = parser.parse_args()

    build(args.config, args.out, args.cache, args.keep_raw)
    return 0


if __name__ == "__main__":
    sys.exit(main())
