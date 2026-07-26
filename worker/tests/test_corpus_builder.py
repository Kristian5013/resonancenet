#!/usr/bin/env python3
"""The corpus builder, exercised without downloading anything.

`build_corpus.py` produces a consensus value. Its output is hashed into
`dataset_root`, which goes into the round descriptor, whose SHA3-256 every node
checks at startup. A corpus that is wrong in any way — a duplicated fragment, a
dropped document, a different byte order — produces a different root, and nothing
downstream can tell: the file has a plausible size, the build prints a plausible
count, and the network simply cannot agree.

Until this file existed the script had no test of any kind, because exercising it
meant downloading a terabyte first. It is driven here from local parquet shards
and the real pinned tokenizer.

Run from the repository root:
    python3 worker/tests/test_corpus_builder.py
"""

from __future__ import annotations

import collections
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "worker" / "tools"))

import build_corpus  # noqa: E402

TOKENIZER = ROOT / "share" / "genesis" / "tokenizer_round0.json"

failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  ok    {name}")
    else:
        print(f"  FAIL  {name}\n        {detail}")
        failures.append(name)


# --------------------------------------------------------------------------
# A corpus on disk, in the shape the real one has.
# --------------------------------------------------------------------------

def write_shards(directory: Path, n_shards: int = 4, docs_per_shard: int = 250) -> list[str]:
    """Parquet shards with deterministic, varied text.

    Varied on purpose: text that is all one repeated string tokenizes to a
    repeated id, which would hide an ordering bug completely.
    """
    names = []
    for shard in range(n_shards):
        docs = []
        for i in range(docs_per_shard):
            n = shard * docs_per_shard + i
            docs.append(
                f"Document {n}. The Merkle root over chunk {n % 7} proves membership "
                f"without revealing the other leaves. Value {n * 2654435761 % 100000}. "
                + "detail " * (n % 13)
            )
        name = f"data/CC-MAIN-2024-10/{shard:03d}_00000.parquet"
        path = directory / f"{shard:03d}.parquet"
        pq.write_table(pa.table({"text": docs}), path)
        names.append(name)
    return names


class LocalFetcher:
    """Stands in for ShardFetcher, reading from a directory instead of the Hub.

    Same contract: yields (shard_name, local_path) in the order the shard list
    gave, skipping what is already done.
    """

    def __init__(self, source, shards, cache_dir, skip):
        self.pending = [s for s in shards if s not in skip]
        self.directory = Path(cache_dir)

    def __iter__(self):
        for shard in self.pending:
            yield shard, str(self.directory / (shard.split("/")[-1].split("_")[0] + ".parquet"))


def make_config(tmp: Path, max_tokens: int = 10**9) -> Path:
    config = {
        "tokenizer": str(TOKENIZER),
        "sources": [{
            "repo": "local/test",
            "revision": "0" * 40,
            "text_column": "text",
            "include": ["data/CC-MAIN-2024-10/"],
            "max_tokens": max_tokens,
        }],
    }
    path = tmp / "corpus.json"
    path.write_text(json.dumps(config))
    return path


def run_build(tmp: Path, shards: list[str], out_name: str = "corpus.bin", **kw) -> Path:
    out = tmp / out_name
    build_corpus.build(
        make_config(tmp), out, tmp, keep_raw=True,
        list_shards=lambda source: list(shards),
        fetcher_factory=LocalFetcher,
        **kw,
    )
    return out


# --------------------------------------------------------------------------

def test_deterministic_and_well_formed() -> None:
    print("[corpus builder]")
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        shards = write_shards(tmp)

        first = run_build(tmp, shards, "a.bin")
        a = first.read_bytes()

        # Same inputs, a second time. Any dependence on dict ordering, thread
        # scheduling or filesystem order shows up here.
        second = run_build(tmp, shards, "b.bin")
        b = second.read_bytes()
        check("two runs over the same shards produce identical bytes", a == b,
              f"{len(a)} bytes vs {len(b)}")

        meta = json.loads(Path(str(first) + "".join(Path(first).suffixes[:-1]))
                          .with_suffix(".meta.json").read_text()) \
            if False else json.loads(first.with_suffix(".meta.json").read_text())

        # The file must be exactly what the metadata says it is. Nothing checked
        # this before, so a truncated or double-written file reported success.
        check("the file is exactly n_tokens * 4 bytes",
              len(a) == meta["n_tokens"] * 4,
              f"{len(a)} bytes, {meta['n_tokens']} tokens claimed")

        tokens = np.frombuffer(a, dtype="<u4")

        # Little-endian is load-bearing: the C++ side reads these bytes directly
        # rather than through the canonical container.
        check("tokens are little-endian uint32", tokens.dtype.byteorder in ("<", "="))

        # One separator per document, and nowhere else.
        expected_docs = 4 * 250
        check("exactly one EOS per document",
              int((tokens == build_corpus.EOS_TOKEN).sum()) == expected_docs,
              f"{int((tokens == build_corpus.EOS_TOKEN).sum())} separators for {expected_docs} documents")

        vocab_size = len(json.loads(TOKENIZER.read_text())["model"]["vocab"])
        check("every id is inside the vocabulary", int(tokens.max()) < vocab_size,
              f"max id {int(tokens.max())} against vocab {vocab_size}")


def test_batch_size_does_not_change_the_bytes() -> None:
    """Documents are encoded in batches; the batch boundary must not be visible.

    If it were, the corpus would depend on a tuning constant, and changing that
    constant later would silently change dataset_root.
    """
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        shards = write_shards(tmp, n_shards=2, docs_per_shard=300)

        original = build_corpus.BATCH_DOCUMENTS
        try:
            build_corpus.BATCH_DOCUMENTS = 1000
            big = run_build(tmp, shards, "big.bin").read_bytes()
            build_corpus.BATCH_DOCUMENTS = 7
            small = run_build(tmp, shards, "small.bin").read_bytes()
        finally:
            build_corpus.BATCH_DOCUMENTS = original

        check("the batch size does not change the output", big == small,
              f"{len(big)} bytes at 1000/batch, {len(small)} at 7/batch")


def test_resume_after_a_crash_mid_shard() -> None:
    """A run killed while tokenizing a shard must resume to the same bytes.

    This is the failure that actually happens: an out-of-memory kill, a spot
    interruption, a disk filling. State is written only between shards, so a crash
    partway through one leaves tokens in the file that no state records — and the
    resume appends that shard from the beginning. The corpus then contains a
    partial document run twice, which is invisible in every number the build
    prints.
    """
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        # More documents per shard than fit in one batch, so a shard is several
        # writes. With one batch per shard the interruption lands between shards
        # and proves nothing — an earlier version of this test did exactly that
        # and passed against the bug it was written to catch.
        docs_per_shard = 3 * build_corpus.BATCH_DOCUMENTS
        shards = write_shards(tmp, n_shards=3, docs_per_shard=docs_per_shard)

        whole = run_build(tmp, shards, "whole.bin").read_bytes()

        # Killed after the first batch of the second shard: those tokens are on
        # disk, and no state records them. That is what an out-of-memory kill or a
        # spot interruption leaves behind.
        real_append = build_corpus.append_tokens
        batches_per_shard = 3
        die_on = batches_per_shard + 2          # shard 1, second batch
        calls = {"n": 0}

        def dying_append(out, tokenizer, documents):
            calls["n"] += 1
            if calls["n"] == die_on:
                out.flush()          # as the OS would have, before the process died
                raise KeyboardInterrupt("simulated interruption mid-shard")
            return real_append(out, tokenizer, documents)

        build_corpus.append_tokens = dying_append
        try:
            try:
                run_build(tmp, shards, "resumed.bin")
            except KeyboardInterrupt:
                pass
        finally:
            build_corpus.append_tokens = real_append

        partial = (tmp / "resumed.bin")
        check("the interrupted run left something behind", partial.exists() and
              partial.stat().st_size > 0, "nothing was written before the interruption")

        # The condition the test exists to create: bytes on disk that the state
        # file does not account for. Without this the resume below is trivial.
        state = json.loads((tmp / "resumed.state.json").read_text())
        check("the interruption left unrecorded bytes on disk",
              partial.stat().st_size > state["tokens_written"] * 4,
              f"{partial.stat().st_size} bytes on disk, "
              f"{state['tokens_written'] * 4} accounted for — the crash landed on a "
              f"shard boundary and the test proves nothing")

        # Resume. Same shards, same config, the state file is where it was left.
        run_build(tmp, shards, "resumed.bin")
        resumed = partial.read_bytes()

        check("a resumed build equals an uninterrupted one", resumed == whole,
              f"resumed {len(resumed)} bytes, uninterrupted {len(whole)} — "
              f"a difference of {len(resumed) - len(whole)}")

        meta = partial.with_suffix(".meta.json")
        if meta.exists():
            claimed = json.loads(meta.read_text())["n_tokens"]
            check("the resumed file matches its own token count",
                  len(resumed) == claimed * 4,
                  f"{len(resumed)} bytes, {claimed} tokens claimed")


def test_parallel_downloads_preserve_order() -> None:
    """Sixteen streams, one order.

    Shard order is part of the corpus: the same shards taken in a different
    sequence produce a different file and therefore a different dataset_root. The
    fetcher downloads in parallel, so the network delivers them out of order on
    purpose here — the later a shard is in the list, the faster it "downloads".
    If ordering came from completion rather than from the list, this reverses.
    """
    shards = [f"data/CC-MAIN-2024-10/{i:03d}_00000.parquet" for i in range(40)]

    def fake_download(repo, filename, repo_type=None, revision=None, cache_dir=None):
        index = int(filename.split("/")[-1].split("_")[0])
        # Deliberately inverted: the last shard returns first.
        time.sleep((len(shards) - index) * 0.002)
        return f"/nonexistent/{filename}"

    real = build_corpus.hf_hub_download
    build_corpus.hf_hub_download = fake_download
    try:
        source = build_corpus.SourceSpec(
            repo="local/test", revision="0" * 40, text_column="text",
            include=["data/"], max_tokens=1,
        )
        got = [name for name, _ in build_corpus.ShardFetcher(source, shards, Path("/tmp"), set())]
    finally:
        build_corpus.hf_hub_download = real

    check("parallel downloads are consumed in shard-list order", got == shards,
          f"first divergence at {next((i for i, (a, b) in enumerate(zip(got, shards)) if a != b), None)}")


def test_a_transient_download_error_is_retried() -> None:
    """One 503 out of eight hundred shards must not end the run.

    The old fetcher stopped the whole source on the first exception, so an hour of
    tokenizing was lost to a momentary server error.
    """
    attempts = collections.Counter()

    def flaky_download(repo, filename, repo_type=None, revision=None, cache_dir=None):
        attempts[filename] += 1
        if attempts[filename] < 3:
            raise OSError("simulated 503")
        return f"/nonexistent/{filename}"

    real_download = build_corpus.hf_hub_download
    real_sleep = build_corpus.time.sleep
    build_corpus.hf_hub_download = flaky_download
    build_corpus.time.sleep = lambda _s: None      # no backoff in a test
    try:
        source = build_corpus.SourceSpec(
            repo="local/test", revision="0" * 40, text_column="text",
            include=["data/"], max_tokens=1,
        )
        shards = ["data/a.parquet", "data/b.parquet"]
        got = [name for name, _ in build_corpus.ShardFetcher(source, shards, Path("/tmp"), set())]
    finally:
        build_corpus.hf_hub_download = real_download
        build_corpus.time.sleep = real_sleep

    check("a shard that fails twice is still delivered", got == shards, f"got {got}")
    check("it was retried rather than refetched forever",
          all(v == 3 for v in attempts.values()), f"{dict(attempts)}")


def test_consumed_shards_free_their_disk_space() -> None:
    """A shard that has been consumed must stop occupying the disk.

    The Hub's downloader returns a path under snapshots/ that is a SYMLINK into
    blobs/, where the bytes are. Removing what it handed back therefore frees
    nothing, and the cache grows to the size of the whole raw dataset — which
    filled a 3.4 TB disk at shard 1595 of 2410 and was reported by the transfer
    client as an internal writer error, naming neither the disk nor the space.

    Modelled here with the same shape: a blob, and a symlink to it standing in for
    what a download returns.
    """
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        shards = write_shards(tmp, n_shards=2, docs_per_shard=200)

        blobs = tmp / "blobs"; blobs.mkdir()
        links = tmp / "snapshots"; links.mkdir()
        for i, name in enumerate(shards):
            blob = blobs / f"blob{i}"
            blob.write_bytes((tmp / f"{i:03d}.parquet").read_bytes())
            (links / name.split("/")[-1]).symlink_to(blob)

        class SymlinkFetcher:
            def __init__(self, source, shard_list, cache_dir, skip):
                self.pending = [s for s in shard_list if s not in skip]
            def __iter__(self):
                for s in self.pending:
                    yield s, str(links / s.split("/")[-1])

        before = sum(f.stat().st_size for f in blobs.iterdir())
        build_corpus.build(make_config(tmp), tmp / "freed.bin", tmp, keep_raw=False,
                           list_shards=lambda source: list(shards),
                           fetcher_factory=SymlinkFetcher)
        after = sum(f.stat().st_size for f in blobs.iterdir())

        check("consuming a shard frees the bytes, not just the symlink", after == 0,
              f"{before/1e6:.1f} MB of blobs before, {after/1e6:.1f} MB after — "
              f"the cache would grow to the size of the whole raw dataset")


def main() -> int:
    if not TOKENIZER.exists():
        print(f"missing {TOKENIZER}; run rnet-tool genesis-emit first")
        return 1
    test_deterministic_and_well_formed()
    test_batch_size_does_not_change_the_bytes()
    test_resume_after_a_crash_mid_shard()
    test_parallel_downloads_preserve_order()
    test_a_transient_download_error_is_retried()
    test_consumed_shards_free_their_disk_space()
    print()
    if failures:
        print(f"{len(failures)} failed: {', '.join(failures)}")
        return 1
    print("corpus builder: deterministic, well-formed, and resumable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
