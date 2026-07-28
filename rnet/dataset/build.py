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


class BuildError(Exception):
    pass


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


def text_from_parquet(path: str, column: str) -> list:
    """Every document in one parquet file, in file order."""
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=[column])
    return table.column(column).to_pylist()


def build(repo_id: str, out_path: str, *, column: str = "text",
          cache_dir: str | None = None, parallel: int = DEFAULT_PARALLEL,
          token: str | None = None, limit_files: int = 0,
          log=print) -> BuildState:
    """Download `repo_id` and append its text to `out_path`, resumably."""
    from huggingface_hub import hf_hub_download, list_repo_files

    out_dir = os.path.dirname(os.path.abspath(out_path)) or "."
    os.makedirs(out_dir, exist_ok=True)
    if cache_dir is None:
        # Beside the output, never in the home directory. See the module note.
        cache_dir = os.path.join(out_dir, "hf-cache")
    os.makedirs(cache_dir, exist_ok=True)

    state = BuildState.load(out_path + ".state")

    # Sorted, and sorted is the specification rather than a convenience.
    files = sorted(f for f in list_repo_files(repo_id, repo_type="dataset",
                                              token=token)
                   if f.endswith(".parquet"))
    if limit_files:
        files = files[:limit_files]
    remaining = [f for f in files if f not in state.files_done]

    log(f"{repo_id}: {len(files)} parquet file(s), {len(remaining)} to go")
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
                                             cache_dir=cache_dir, token=token)
            except Exception as exc:                      # noqa: BLE001
                last = exc
                time.sleep(min(BACKOFF_CAP_S, 2.0 ** attempt))
        raise BuildError(f"corpus: {name} failed after {MAX_ATTEMPTS} attempts: {last}")

    started = time.monotonic()
    at_start = state.bytes_written

    with ThreadPoolExecutor(max_workers=parallel) as pool, \
            open(out_path, "ab") as out:
        # Submitted in order and consumed in order. `map` preserves ordering
        # regardless of which download finishes first, which is exactly the
        # property the root depends on.
        for name, local in pool.map(fetch, remaining):
            try:
                documents = text_from_parquet(local, column)
            except Exception as exc:                      # noqa: BLE001
                raise BuildError(f"corpus: {name} did not parse: {exc}") from exc

            written = 0
            for doc in documents:
                if not doc:
                    continue
                encoded = doc.encode("utf-8")
                out.write(encoded)
                out.write(SEPARATOR)
                written += len(encoded) + len(SEPARATOR)
            out.flush()
            os.fsync(out.fileno())

            state.bytes_written += written
            state.documents += len(documents)
            state.files_done.append(name)
            state.save()
            freed = remove_download(local)

            elapsed = max(1e-6, time.monotonic() - started)
            rate = (state.bytes_written - at_start) / elapsed
            done, total = len(state.files_done), len(files)
            left = (total - done) * (elapsed / max(1, done - 0)) if done else 0
            log(f"[{done:>6}/{total}] {state.bytes_written / 2**40:6.3f} TiB  "
                f"{rate / 2**20:6.1f} MiB/s  freed {freed / 2**20:5.0f} MiB  "
                f"{left / 3600:5.1f}h left")

    log(f"done: {state.bytes_written:,} bytes, {state.documents:,} documents")
    return state


def free_space(path: str) -> int:
    return shutil.disk_usage(path).free
