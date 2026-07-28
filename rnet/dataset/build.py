"""Turning a Hugging Face dataset into the one file the corpus is addressed over.

The corpus is raw UTF-8: documents separated by a blank line, in a fixed order,
in one file. Everything downstream — the chunk boundaries, the Merkle tree, the
root a round pins — is a function of those bytes, so the two things this module
must get right are WHICH bytes and IN WHICH ORDER.

ORDER IS PART OF THE ARTIFACT. Parquet files are downloaded concurrently because
that is the only way to saturate the link, and appended STRICTLY in sorted name
order regardless of which finished first. Append in completion order and the
same dataset produces a different root on every machine, which would be found
much later and by somebody else.

THE THREE THINGS THAT WENT WRONG LAST TIME, all of them operational rather than
clever:

  * The cache filled the root disk. Hugging Face caches to ~/.cache by default;
    on a box whose root volume is 40 GB and whose data volume is 9 TB, that is
    a build that dies at 8% with no disk left. The cache goes beside the output.

  * Deleting a downloaded file deleted the symlink, not the file. hf_hub_download
    returns a path inside snapshots/ that points into blobs/; unlinking it frees
    nothing. 1,598 blobs and 3.4 TB accumulated before the disk ran out, and the
    symptom was an unrelated-looking error from the writer.

  * A failed resume corrupted the output. The state file records bytes written,
    and the output is truncated to exactly that before appending — otherwise a
    partial record from an interrupted write is silently inside the corpus and
    the root is wrong in a way nothing detects.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

SEPARATOR = b"\n\n"

# How many parquet files to have in flight. Past a handful the bottleneck is the
# remote end rather than this one, and every file in flight is disk held.
DEFAULT_PARALLEL = 8

# Retries per file. Hugging Face rate-limits and times out under sustained load,
# and a build that gives up on one file of twenty thousand has wasted a day.
MAX_ATTEMPTS = 12
BACKOFF_CAP_S = 300.0

# Rows read from a parquet file at a time. Small enough that one batch is a
# bounded allocation regardless of how large the file is, large enough that the
# per-batch overhead disappears.
BATCH_ROWS = 4096


class BuildError(Exception):
    pass


# Building a corpus is the one job here that reaches outside the standard
# library and numpy, and neither package was declared anywhere, so the
# documented install could not run the documented command — it died on a raw
# ModuleNotFoundError from inside a helper.
_MISSING = ("corpus: building needs `{name}`, which is not installed.\n  "
            "pip install -e '.[corpus]'   (huggingface_hub and pyarrow)\n"
            "Only building needs it. Reading a corpus you already have does "
            "not.")


@dataclass
class BuildState:
    """What has been written, so an interrupted build resumes exactly."""

    path: str
    bytes_written: int = 0
    files_done: list = field(default_factory=list)
    documents: int = 0

    @classmethod
    def load(cls, path: str) -> "BuildState":
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, ValueError):
            return cls(path=path)
        return cls(path=path, bytes_written=int(data.get("bytes_written", 0)),
                   files_done=list(data.get("files_done", [])),
                   documents=int(data.get("documents", 0)))

    def save(self) -> None:
        """Atomically, because a state file torn by a crash is worse than none:
        it would claim bytes the output does not have."""
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"bytes_written": self.bytes_written,
                       "files_done": self.files_done,
                       "documents": self.documents}, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)


def remove_download(path: str) -> int:
    """Delete a downloaded file AND the blob it points at. Returns bytes freed.

    hf_hub_download hands back a path under snapshots/ which is a symlink into
    blobs/. Unlinking it frees nothing, and the disk fills while the build looks
    like it is cleaning up after itself.
    """
    freed = 0
    try:
        target = os.path.realpath(path)
        if os.path.exists(target):
            freed = os.path.getsize(target)
            os.remove(target)
        if os.path.islink(path) or os.path.exists(path):
            os.remove(path)
    except OSError:
        pass
    return freed


def append_parquet(path: str, column: str, out) -> tuple[int, int]:
    """Stream one parquet file into `out`. Returns (bytes written, documents).

    Batched rather than `read_table(...).to_pylist()`, which was the first
    version and was the bottleneck: it pulls a 2.3 GB file wholly into memory
    and then builds half a million Python string objects from it. On a 30 GB box
    with eight downloads also buffering, that is memory pressure severe enough
    to stall the network — the incoming rate measured ZERO while a file was
    being converted, and the pipeline looked like a slow download when it was
    really a slow reader.
    """
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise BuildError(_MISSING.format(name=exc.name)) from exc

    written = documents = 0
    reader = pq.ParquetFile(path)
    for batch in reader.iter_batches(batch_size=BATCH_ROWS, columns=[column]):
        for value in batch.column(0):
            text = value.as_py()
            if not text:
                continue
            encoded = text.encode("utf-8")
            out.write(encoded)
            out.write(SEPARATOR)
            written += len(encoded) + len(SEPARATOR)
            documents += 1
    return written, documents


def build(repo_id: str, out_path: str, *, column: str = "text",
          cache_dir: str | None = None, parallel: int = DEFAULT_PARALLEL,
          token: str | None = None, limit_files: int = 0,
          revision: str | None = None, include: tuple = (), log=print) -> BuildState:
    """Download `repo_id` and append its text to `out_path`, resumably.

    `revision` NAMES THE SNAPSHOT and is the difference between a corpus root
    that can be checked and one that merely happened. Without it both calls
    below track the repository's moving `main`, so a build run a month after
    the pin silently sees a different file list and different contents, and
    produces a root that will never match — with no way to tell that apart from
    a bug in this code. FineWeb-Edu grew from 4.52 TB to 5.84 TB between the
    pin and 2026-07-28, which is exactly how this was noticed.

    Pass the commit sha a network pinned. `None` means "whatever is there now",
    which is honest for a first build that is about to define a new pin.

    `include` KEEPS ONLY PATHS UNDER THESE PREFIXES, and a dataset laid out like
    FineWeb-Edu needs it. That repository holds 3,036 parquet files: 2,410 under
    `data/`, and 626 under `sample/` which are COPIES of subsets of the first
    2,410. Taking everything appends about two terabytes of text the corpus
    already contains — a root that matches nothing, and training data in which
    some documents appear twice. Nothing here could detect that afterwards: a
    duplicate is a perfectly well-formed document.

    It also explains a discrepancy that looked like the dataset moving under us.
    `data/` alone is 4.52 TB of parquet, the whole repository is 5.84 TB, and
    the difference is the samples rather than a year of growth.

    Several prefixes are allowed because the slice worth training on is rarely
    one directory. FineWeb-Edu is 110 crawls of about 45 GiB each and FinePDFs
    is 1,748 languages, so "a hundred gigabytes of recent English" is two or
    three prefixes, not one. The ORDER IS STILL THE SORTED ORDER OF THE PATHS,
    not the order the prefixes were given — otherwise the same set of prefixes
    written in a different order would produce a different root.
    """
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError as exc:
        raise BuildError(_MISSING.format(name=exc.name)) from exc

    out_dir = os.path.dirname(os.path.abspath(out_path)) or "."
    os.makedirs(out_dir, exist_ok=True)
    if cache_dir is None:
        # Beside the output, never in the home directory. See the module note.
        cache_dir = os.path.join(out_dir, "hf-cache")
    os.makedirs(cache_dir, exist_ok=True)

    state = BuildState.load(out_path + ".state")

    # Sorted, and sorted is the specification rather than a convenience.
    every = sorted(f for f in list_repo_files(repo_id, repo_type="dataset",
                                              revision=revision, token=token)
                   if f.endswith(".parquet"))
    prefixes = (include,) if isinstance(include, str) and include else tuple(include)
    files = [f for f in every if f.startswith(prefixes)] if prefixes else every
    dropped = len(every) - len(files)
    if limit_files:
        files = files[:limit_files]
    remaining = [f for f in files if f not in state.files_done]

    log(f"{repo_id}@{revision or 'main (UNPINNED — the root this produces names no snapshot)'}: "
        f"{len(files)} parquet file(s), {len(remaining)} to go"
        + (f" ({dropped} outside {', '.join(prefixes)} skipped)" if dropped else ""))
    if prefixes and not files:
        raise BuildError(
            f"corpus: nothing in {repo_id} is under {', '.join(prefixes)}. "
            + (f"The first paths are {', '.join(every[:3])}" if every
               else "The repo has no parquet files at all."))

    # A file already written that the filter now excludes is text inside the
    # corpus that this build would not choose again — silently different from
    # what the same command produces on an empty directory.
    stale = [f for f in state.files_done if f not in set(files)]
    if stale:
        raise BuildError(
            f"corpus: {out_path} already contains {len(stale)} file(s) that "
            f"{', '.join(prefixes)} excludes, the first being {stale[0]}.\n"
            f"Resuming would leave them in and the root would describe neither "
            f"filter. Start this output again, or drop --include.")
    if state.bytes_written:
        log(f"resuming at {state.bytes_written / 2**40:.2f} TiB, "
            f"{state.documents:,} documents")

    # Truncate to exactly what the state claims. Anything past it is a partial
    # write from an interrupted run, and leaving it in would put a torn record
    # inside the corpus where nothing would ever notice.
    with open(out_path, "ab") as f:
        pass
    if os.path.getsize(out_path) != state.bytes_written:
        log(f"truncating {os.path.getsize(out_path)} -> {state.bytes_written}")
        with open(out_path, "r+b") as f:
            f.truncate(state.bytes_written)

    def fetch(name: str) -> tuple:
        last = None
        for attempt in range(MAX_ATTEMPTS):
            try:
                return name, hf_hub_download(repo_id, name, repo_type="dataset",
                                             revision=revision,
                                             cache_dir=cache_dir, token=token)
            except Exception as exc:                      # noqa: BLE001
                last = exc
                time.sleep(min(BACKOFF_CAP_S, 2.0 ** attempt))
        raise BuildError(f"corpus: {name} failed after {MAX_ATTEMPTS} attempts: {last}")

    started = time.monotonic()
    at_start = state.bytes_written

    with ThreadPoolExecutor(max_workers=parallel) as pool, \
            open(out_path, "ab") as out:
        # A BOUNDED window of downloads in flight, not `pool.map` over all of
        # them. map submits every task at once, so downloads race ahead of the
        # reader and every finished file sits on disk waiting — measured at 64
        # files and 152 GB of cache before this was noticed, on a build that
        # would have kept growing until the volume filled.
        #
        # Order is still exactly the submission order, which is what the corpus
        # root depends on: results are consumed from the head of the window.
        window: list = []
        cursor = 0
        depth = parallel + 2

        while cursor < len(remaining) or window:
            while len(window) < depth and cursor < len(remaining):
                window.append(pool.submit(fetch, remaining[cursor]))
                cursor += 1

            name, local = window.pop(0).result()
            try:
                written, documents = append_parquet(local, column, out)
            except Exception as exc:                      # noqa: BLE001
                raise BuildError(f"corpus: {name} did not parse: {exc}") from exc
            out.flush()
            os.fsync(out.fileno())

            state.bytes_written += written
            state.documents += documents
            state.files_done.append(name)
            state.save()
            freed = remove_download(local)

            elapsed = max(1e-6, time.monotonic() - started)
            gained = state.bytes_written - at_start
            done, total = len(state.files_done), len(files)
            todo = total - done
            log(f"[{done:>6}/{total}] {state.bytes_written / 2**40:6.3f} TiB  "
                f"{gained / elapsed / 2**20:6.1f} MiB/s  "
                f"freed {freed / 2**20:5.0f} MiB  "
                f"{todo * elapsed / max(1, done) / 3600:5.1f}h left")

    log(f"done: {state.bytes_written:,} bytes, {state.documents:,} documents")
    return state


def free_space(path: str) -> int:
    return shutil.disk_usage(path).free
