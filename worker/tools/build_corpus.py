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
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_files
from tokenizers import Tokenizer

# Token 0 in this vocabulary. Appended after every document and nowhere else.
EOS_TOKEN = 0

# Documents encoded per call. Large enough that the Rust tokenizer's parallelism
# pays off, small enough that a batch of long documents does not spike memory.
BATCH_DOCUMENTS = 1000


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
    """Loads the tokenizer and refuses one that is not the pinned one.

    A corpus built with a different tokenizer is a different corpus wearing the
    right name: the token ids mean other strings, and nothing downstream can tell.
    """
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
    print(f"tokenizer {actual[:16]}… vocab {json.loads(raw)['model']['vocab_size'] if 'vocab_size' in raw[:200].decode('utf-8', 'ignore') else len(json.loads(raw)['model']['vocab'])}")
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


def build(config_path: Path, out_path: Path, cache_dir: Path, keep_raw: bool) -> None:
    config = json.loads(config_path.read_text())
    sources = [SourceSpec(**s) for s in config["sources"]]

    tokenizer = load_tokenizer(Path(config["tokenizer"]), config.get("tokenizer_hash"))

    state_path = out_path.with_suffix(".state.json")
    state = BuildState(**json.loads(state_path.read_text())) if state_path.exists() else BuildState()
    if state.tokens_written:
        print(f"resuming: {state.tokens_written:,} tokens already written, "
              f"{len(state.completed_shards)} shards done")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Append, so a resumed run continues the same file rather than starting a
    # second one. Truncating here would silently discard days of work.
    mode = "ab" if state.tokens_written else "wb"
    started = time.time()

    with open(out_path, mode) as out:
        for source_index, source in enumerate(sources):
            if source_index < state.source_index:
                continue
            state.source_index = source_index
            written_here = state.source_tokens.get(source.repo, 0)

            shards = shard_list(source)
            print(f"\n{source.repo} @ {source.revision[:12]}: {len(shards)} shards, "
                  f"target {source.max_tokens:,} tokens")

            for shard in shards:
                if shard in state.completed_shards:
                    continue
                if written_here >= source.max_tokens:
                    print(f"  reached {written_here:,} tokens; moving on")
                    break

                local = hf_hub_download(
                    source.repo, shard, repo_type="dataset",
                    revision=source.revision, cache_dir=str(cache_dir),
                )
                table = pq.read_table(local)

                batch: list[str] = []
                shard_tokens = 0
                for text in documents_in(table, source):
                    batch.append(text)
                    if len(batch) >= BATCH_DOCUMENTS:
                        n, raw = append_tokens(out, tokenizer, batch)
                        shard_tokens += n
                        state.raw_bytes_read += raw
                        batch.clear()
                if batch:
                    n, raw = append_tokens(out, tokenizer, batch)
                    shard_tokens += n
                    state.raw_bytes_read += raw

                written_here += shard_tokens
                state.tokens_written += shard_tokens
                state.source_tokens[source.repo] = written_here
                state.completed_shards.append(shard)

                # Deleted as consumed, so peak disk is the output plus one shard
                # rather than the whole raw dataset.
                if not keep_raw:
                    try:
                        os.remove(local)
                    except OSError:
                        pass

                out.flush()
                state_path.write_text(json.dumps(asdict(state)))

                elapsed = time.time() - started
                ratio = state.raw_bytes_read / max(state.tokens_written, 1)
                print(f"  {shard.split('/')[-1]}: +{shard_tokens:,} → {state.tokens_written:,} "
                      f"({ratio:.2f} raw bytes/token, {state.tokens_written / max(elapsed, 1):,.0f} tok/s)")

    ratio = state.raw_bytes_read / max(state.tokens_written, 1)
    size = out_path.stat().st_size
    print(f"\n{state.tokens_written:,} tokens, {size / 1e9:.1f} GB on disk")
    print(f"{ratio:.3f} raw bytes per token with THIS tokenizer")
    # The number that sizes a disk. Published token counts use other tokenizers,
    # and a larger vocabulary yields fewer tokens for the same text — so a plan
    # built on someone else's count is a plan built on someone else's vocabulary.
    print(f"→ {ratio * 1e12 / 1e12:.2f} TB of raw text per trillion tokens, "
          f"{4e12 / 1e12:.0f} TB on disk per trillion tokens")

    meta = {
        "n_tokens": state.tokens_written,
        "dtype": "uint32",
        "byte_order": "little-endian",
        "eos": EOS_TOKEN,
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
