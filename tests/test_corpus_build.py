"""Assembling a corpus from a remote dataset.

The build is the only place where bytes enter the protocol from outside, so
what it selects and in what order IS the artifact. A root computed over the
wrong file list is not detectably wrong later — every document in it is a
perfectly well-formed document.

These tests drive `build` against a fake Hugging Face, because the properties
worth pinning are about selection, ordering and resume, and none of them need
the network to exercise.
"""

import json
import os
import shutil
import tempfile
import unittest

from rnet.dataset import build as B


class FakeHub:
    """Enough of huggingface_hub to answer the two calls `build` makes.

    Files are (path -> list of documents). Written as real parquet, because the
    reader is the code under test as much as the selection is.
    """

    def __init__(self, files: dict, revision_files: dict | None = None):
        self.files = files
        self.revision_files = revision_files or {}
        self.asked_revisions = []
        self.fetched = []

    def install(self, case):
        import huggingface_hub

        def list_repo_files(repo_id, repo_type=None, revision=None, token=None):
            self.asked_revisions.append(revision)
            return list((self.revision_files.get(revision) or self.files).keys())

        def hf_hub_download(repo_id, name, repo_type=None, revision=None,
                            cache_dir=None, token=None):
            self.fetched.append(name)
            table = (self.revision_files.get(revision) or self.files)[name]
            path = os.path.join(cache_dir, name.replace("/", "_"))
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            _write_parquet(path, table)
            return path

        case.enterContext(_patched(huggingface_hub, "list_repo_files", list_repo_files))
        case.enterContext(_patched(huggingface_hub, "hf_hub_download", hf_hub_download))
        return self


class _patched:
    def __init__(self, module, name, value):
        self.module, self.name, self.value = module, name, value

    def __enter__(self):
        self.old = getattr(self.module, self.name, None)
        setattr(self.module, self.name, self.value)

    def __exit__(self, *exc):
        if self.old is None:
            delattr(self.module, self.name)
        else:
            setattr(self.module, self.name, self.old)


def _write_parquet(path: str, documents: list) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    pq.write_table(pa.table({"text": pa.array(documents, type=pa.string())}), path)


class BuildTests(unittest.TestCase):

    def setUp(self):
        try:
            import huggingface_hub, pyarrow          # noqa: F401
        except ImportError:
            self.skipTest("needs the corpus extra: pip install -e '.[corpus]'")
        self.dir = tempfile.mkdtemp(prefix="rnet-build-")
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        self.out = os.path.join(self.dir, "corpus.txt")

    def run_build(self, files, **kw):
        hub = FakeHub(files, kw.pop("revision_files", None)).install(self)
        state = B.build("fake/repo", self.out, cache_dir=os.path.join(self.dir, "c"),
                        parallel=2, log=lambda *a: None, **kw)
        return state, hub

    def text(self) -> bytes:
        with open(self.out, "rb") as f:
            return f.read()

    # -- what gets in ---------------------------------------------------------

    def test_ItAppendsInSortedOrderNotCompletionOrder(self):
        """Order is part of the artifact: append as downloads finish and the
        same dataset gives a different root on every machine."""
        state, _ = self.run_build({
            "data/c.parquet": ["gamma"],
            "data/a.parquet": ["alpha"],
            "data/b.parquet": ["beta"],
        })
        self.assertEqual(self.text(), b"alpha\n\nbeta\n\ngamma\n\n")
        self.assertEqual(state.documents, 3)
        self.assertEqual(state.bytes_written, len(self.text()))

    def test_IncludeKeepsOnlyThatPrefix(self):
        """FineWeb-Edu ships 2,410 files under data/ and 626 under sample/ that
        are copies of subsets of the first 2,410. Taking both appends about two
        terabytes the corpus already contains — a root matching nothing, and
        documents the model would see twice."""
        state, hub = self.run_build({
            "data/a.parquet": ["kept"],
            "data/b.parquet": ["also kept"],
            "sample/10BT/a.parquet": ["a copy of data/a"],
            "sample/100BT/b.parquet": ["a copy of data/b"],
        }, include="data/")
        self.assertEqual(self.text(), b"kept\n\nalso kept\n\n")
        self.assertEqual(state.documents, 2)
        # And the excluded ones were never even fetched — the cost is skipped,
        # not merely the bytes.
        self.assertEqual(sorted(hub.fetched), ["data/a.parquet", "data/b.parquet"])

    def test_AnIncludeThatMatchesNothingIsRefused(self):
        with self.assertRaises(B.BuildError) as caught:
            self.run_build({"data/a.parquet": ["x"]}, include="corpus/")
        self.assertIn("corpus/", str(caught.exception))

    def test_TheRevisionIsWhatIsAskedFor(self):
        """Without it both calls track a moving branch, so the root names no
        snapshot and a rebuild a month later differs for reasons nothing can
        distinguish from a bug."""
        state, hub = self.run_build(
            {"data/a.parquet": ["new"]},
            revision="abc123",
            revision_files={"abc123": {"data/a.parquet": ["as it was pinned"]}})
        self.assertEqual(self.text(), b"as it was pinned\n\n")
        self.assertEqual(hub.asked_revisions, ["abc123"])

    def test_EmptyDocumentsAreSkippedNotWritten(self):
        """A separator with nothing before it would be a zero-length document,
        which changes every boundary after it."""
        state, _ = self.run_build({"data/a.parquet": ["one", "", "two", None]})
        self.assertEqual(self.text(), b"one\n\ntwo\n\n")
        self.assertEqual(state.documents, 2)

    def test_LimitTakesThePrefixOfTheSortedList(self):
        state, _ = self.run_build({f"data/{c}.parquet": [c] for c in "abcde"},
                                  limit_files=2)
        self.assertEqual(self.text(), b"a\n\nb\n\n")

    # -- resume ---------------------------------------------------------------

    def test_ARepeatedBuildAddsNothing(self):
        files = {"data/a.parquet": ["one"], "data/b.parquet": ["two"]}
        self.run_build(files)
        first = self.text()
        state, hub = self.run_build(files)
        self.assertEqual(self.text(), first)
        self.assertEqual(hub.fetched, [])

    def test_AnInterruptedWriteIsTruncatedBeforeAppending(self):
        """The state file is the authority on what is really in there. A
        partial record from a killed process would otherwise sit inside the
        corpus, and nothing downstream could tell."""
        files = {"data/a.parquet": ["one"], "data/b.parquet": ["two"]}
        self.run_build(files)
        with open(self.out, "ab") as f:
            f.write(b"half a document, no separator")
        state, _ = self.run_build(files)
        self.assertEqual(self.text(), b"one\n\ntwo\n\n")
        self.assertEqual(state.bytes_written, len(self.text()))

    def test_ResumingUnderANarrowerFilterIsRefused(self):
        """Because the output already holds text the new filter would not
        choose, and the resulting root would describe neither run."""
        files = {"data/a.parquet": ["kept"], "sample/a.parquet": ["a copy"]}
        self.run_build(files)
        with self.assertRaises(B.BuildError) as caught:
            self.run_build(files, include="data/")
        self.assertIn("sample/a.parquet", str(caught.exception))

    def test_TheStateSurvivesACrashDuringSave(self):
        """Written to a temporary file and renamed, because a torn state file
        claims bytes the output does not have."""
        self.run_build({"data/a.parquet": ["one"]})
        with open(self.out + ".state") as f:
            saved = json.load(f)
        self.assertEqual(saved["files_done"], ["data/a.parquet"])
        self.assertEqual([f for f in os.listdir(self.dir) if f.endswith(".tmp")], [])

    # -- housekeeping ---------------------------------------------------------

    def test_TheDownloadedBlobIsFreedNotJustItsSymlink(self):
        """hf_hub_download hands back a path under snapshots/ pointing into
        blobs/. Unlinking it frees nothing — 3.4 TB accumulated that way once,
        and the symptom was an unrelated-looking error from the writer."""
        blob = os.path.join(self.dir, "blob.bin")
        with open(blob, "wb") as f:
            f.write(b"x" * 4096)
        link = os.path.join(self.dir, "link.parquet")
        os.symlink(blob, link)

        freed = B.remove_download(link)
        self.assertEqual(freed, 4096)
        self.assertFalse(os.path.exists(blob))
        self.assertFalse(os.path.lexists(link))


if __name__ == "__main__":
    unittest.main(verbosity=2)
