"""Reading and indexing a corpus.

The property everything rests on: two implementations of the boundary rule
would be two chances to disagree about what chunk seven is, discovered long
after everyone has trained on different text. So there is one implementation
and these tests pin its edges.
"""

import json
import os
import shutil
import tempfile
import unittest

from rnet.crypto import merkle
from rnet.dataset.corpus import (SCAN_WINDOW, SEPARATOR, CorpusError,
                                 LocalCorpus, derive_offsets)
from rnet.dataset.index import CorpusIndex, build_index, proof_for, verify_chunk
from rnet.dataset.scheduler import window_seed


class Tokenizer:
    """Bytes as tokens. Enough to exercise the window path without pulling in
    a real tokenizer, and the vocabulary is honest about being 256."""

    class Encoding:
        def __init__(self, ids):
            self.ids = ids

    def encode(self, text: str):
        return self.Encoding(list(text.encode("utf-8")))


class CorpusFixture(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="rnet-corpus-")
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)

    def write(self, documents, name="corpus.txt") -> str:
        path = os.path.join(self.dir, name)
        with open(path, "wb") as f:
            for doc in documents:
                f.write(doc if isinstance(doc, bytes) else doc.encode())
                f.write(SEPARATOR)
        return path

    def documents(self, n: int, size: int = 400) -> list:
        # Varying lengths on purpose: equal ones put every boundary at a regular
        # interval, which is exactly the case that would hide a bug in deriving
        # them.
        #
        # And varying CONTENT, which is not decoration. Filling documents with a
        # repeated character made two genuinely different windows compare equal,
        # so a test meant to prove that different seeds draw different text
        # passed or failed on the filler rather than on the schedule.
        out = []
        for i in range(n):
            length = size + (i * 137) % (size * 3)
            body = "".join(chr(97 + ((i * 31 + j * 7) % 26)) for j in range(length))
            out.append(f"doc{i} {body}")
        return out


class BoundaryTests(CorpusFixture):

    def test_EveryBoundaryFollowsASeparator(self):
        path = self.write(self.documents(300))
        with open(path, "rb") as f:
            raw = f.read()
        offsets = derive_offsets(path, 4096)

        self.assertEqual(offsets[0], 0)
        self.assertEqual(offsets[-1], len(raw))
        self.assertGreater(len(offsets), 2)
        for i in range(1, len(offsets) - 1):
            at = offsets[i]
            self.assertEqual(raw[at - 2:at], SEPARATOR,
                             f"boundary {i} at {at} does not follow a blank line")

    def test_NoChunkIsShorterThanTheTargetExceptTheLast(self):
        path = self.write(self.documents(300))
        offsets = derive_offsets(path, 4096)
        for i in range(len(offsets) - 2):
            self.assertGreaterEqual(offsets[i + 1] - offsets[i], 4096, i)

    def test_ItIsTheSameOnEveryPass(self):
        path = self.write(self.documents(200))
        self.assertEqual(derive_offsets(path, 4096), derive_offsets(path, 4096))

    def test_ADifferentTargetIsADifferentChunking(self):
        path = self.write(self.documents(200))
        self.assertNotEqual(derive_offsets(path, 4096),
                            derive_offsets(path, 8192))

    def test_ASeparatorStraddlingTheScanWindowIsFound(self):
        """The edge that matters and would otherwise be silent.

        The scan reads in windows and steps back one byte between them, because
        a separator whose two bytes fall either side of a window edge is
        invisible to a naive `find`. Missing one shifts every boundary after it,
        so two implementations agree on the first chunks and disagree on the
        rest.
        """
        # Place a separator exactly across the window boundary: the first byte
        # of SEPARATOR is the last byte of one read.
        head = b"a" * (SCAN_WINDOW - 1)
        path = os.path.join(self.dir, "edge.txt")
        with open(path, "wb") as f:
            f.write(b"lead" + SEPARATOR)          # a boundary to start from
            f.write(head)
            f.write(SEPARATOR)                     # straddles the edge
            f.write(b"tail" * 100)
            f.write(SEPARATOR)

        offsets = derive_offsets(path, 8)
        with open(path, "rb") as f:
            raw = f.read()
        for i in range(1, len(offsets) - 1):
            self.assertEqual(raw[offsets[i] - 2:offsets[i]], SEPARATOR,
                             f"boundary {i} missed the straddling separator")
        # And it really did span the edge, or the test proves nothing.
        self.assertIn(len(b"lead") + 2 + SCAN_WINDOW - 1, range(len(raw)))

    def test_AnEmptyFileIsRefused(self):
        path = os.path.join(self.dir, "empty.txt")
        open(path, "wb").close()
        with self.assertRaises(CorpusError):
            derive_offsets(path, 4096)

    def test_AFileWithNoSeparatorIsOneChunk(self):
        path = os.path.join(self.dir, "flat.txt")
        with open(path, "wb") as f:
            f.write(b"x" * 100_000)
        self.assertEqual(derive_offsets(path, 4096), [0, 100_000])


class LocalCorpusTests(CorpusFixture):

    def corpus(self, n=200) -> LocalCorpus:
        path = self.write(self.documents(n))
        return LocalCorpus.open(path, 4096, Tokenizer())

    def test_ChunksTileTheFileExactly(self):
        c = self.corpus()
        joined = b"".join(c.get(i) for i in range(c.n_chunks))
        with open(c.path, "rb") as f:
            self.assertEqual(joined, f.read())

    def test_AnOutOfRangeChunkIsRefused(self):
        c = self.corpus()
        with self.assertRaises(CorpusError):
            c.get(c.n_chunks)
        with self.assertRaises(CorpusError):
            c.get(-1)

    def test_AWindowComesFromTheSeedAndNothingElse(self):
        c = self.corpus()
        seed = window_seed(bytes(32), 0, 7, 1, 0)
        first = c.window_for_seed(seed, 65)
        self.assertEqual(len(first), 65)
        self.assertEqual(c.window_for_seed(seed, 65), first)

        other = window_seed(bytes(32), 0, 8, 1, 0)
        self.assertNotEqual(c.window_for_seed(other, 65), first)

    def test_TheTokenCacheDoesNotChangeWhatIsRead(self):
        c = self.corpus()
        c.cache_chunks = 2
        seeds = [window_seed(bytes(32), 0, w, 1, 0) for w in range(12)]
        first = [c.window_for_seed(s, 33) for s in seeds]
        self.assertEqual([c.window_for_seed(s, 33) for s in seeds], first)
        self.assertLessEqual(len(c._tokens), 2)

    def test_SuppliedOffsetsAreUsedRatherThanRescanned(self):
        """Scanning seven terabytes is hours; a node that has already done it
        passes them back in."""
        c = self.corpus()
        again = LocalCorpus.open(c.path, 4096, Tokenizer(), offsets=c.offsets)
        self.assertEqual(again.offsets, c.offsets)
        self.assertEqual(again.n_chunks, c.n_chunks)

    def test_WithoutATokenizerThereAreNoTokens(self):
        path = self.write(self.documents(20))
        c = LocalCorpus.open(path, 4096, tokenizer=None)
        c.get(0)
        with self.assertRaises(CorpusError):
            c.tokens_for_chunk(0)


class IndexTests(CorpusFixture):

    def test_TheIndexUsesTheSameBoundariesAsTheReader(self):
        """One rule, one implementation. The indexer calls the reader's scan
        rather than repeating it, and this is what would catch a future edit
        that forgot."""
        path = self.write(self.documents(300))
        index = build_index(path, 4096)
        reader = LocalCorpus.open(path, 4096)
        self.assertEqual(index.offsets, reader.offsets)
        self.assertEqual(index.n_chunks, reader.n_chunks)

    def test_TheRootIsOverTheChunksTheReaderReturns(self):
        path = self.write(self.documents(120))
        index = build_index(path, 4096)
        reader = LocalCorpus.open(path, 4096)
        leaves = [merkle.leaf_hash(reader.get(i)) for i in range(reader.n_chunks)]
        self.assertEqual(merkle.root(leaves), index.root)

    def test_EveryChunkProvesAgainstTheRoot(self):
        path = self.write(self.documents(60))
        index = build_index(path, 4096)
        reader = LocalCorpus.open(path, 4096)
        for i in range(index.n_chunks):
            proof = proof_for(index, i)
            self.assertTrue(verify_chunk(reader.get(i), proof, index.root), i)

    def test_ATamperedChunkDoesNotProve(self):
        path = self.write(self.documents(40))
        index = build_index(path, 4096)
        reader = LocalCorpus.open(path, 4096)
        proof = proof_for(index, 3)
        tampered = bytearray(reader.get(3))
        tampered[0] ^= 0x01
        self.assertFalse(verify_chunk(bytes(tampered), proof, index.root))

    def test_ChangingOneByteChangesTheRoot(self):
        path = self.write(self.documents(50))
        before = build_index(path, 4096).root
        with open(path, "r+b") as f:
            f.seek(10)
            f.write(b"Z")
        self.assertNotEqual(build_index(path, 4096).root, before)

    def test_TheWorkerCountDoesNotChangeTheRoot(self):
        path = self.write(self.documents(80))
        self.assertEqual(build_index(path, 4096, workers=1).root,
                         build_index(path, 4096, workers=8).root)

    # -- the cache is a cache -------------------------------------------------

    def test_AnIndexRoundTripsThroughDisk(self):
        path = self.write(self.documents(50))
        index = build_index(path, 4096)
        saved = os.path.join(self.dir, "corpus.rnidx")
        index.save(saved)
        back = CorpusIndex.load(saved, path, 4096)
        self.assertIsNotNone(back)
        self.assertEqual(back.root, index.root)
        self.assertEqual(back.offsets, index.offsets)

    def test_AnIndexForADifferentFileIsRejected(self):
        """It is a cache of a derivation, never a source of truth. One that
        could disagree with the corpus and win would be a way to make two nodes
        train on different text while agreeing on a root."""
        path = self.write(self.documents(50))
        saved = os.path.join(self.dir, "corpus.rnidx")
        build_index(path, 4096).save(saved)

        with open(path, "ab") as f:
            f.write(b"one more document\n\n")
        self.assertIsNone(CorpusIndex.load(saved, path, 4096))

    def test_AnIndexAtADifferentTargetIsRejected(self):
        path = self.write(self.documents(50))
        saved = os.path.join(self.dir, "corpus.rnidx")
        build_index(path, 4096).save(saved)
        self.assertIsNone(CorpusIndex.load(saved, path, 8192))

    def test_ACorruptIndexIsRebuiltNotFatal(self):
        path = self.write(self.documents(20))
        saved = os.path.join(self.dir, "corpus.rnidx")
        with open(saved, "w") as f:
            f.write("not json at all")
        self.assertIsNone(CorpusIndex.load(saved, path, 4096))

    def test_SavingIsAtomic(self):
        path = self.write(self.documents(20))
        saved = os.path.join(self.dir, "corpus.rnidx")
        build_index(path, 4096).save(saved)
        self.assertEqual([f for f in os.listdir(self.dir) if f.endswith(".tmp")], [])

    def test_TheManifestDescribesWhatWasIndexed(self):
        path = self.write(self.documents(80))
        index = build_index(path, 4096)
        manifest = index.manifest(tokenizer_hash=bytes([7]) * 32)
        from rnet.consensus.objects import DatasetManifest
        back = DatasetManifest.from_container(manifest.to_container())
        self.assertEqual(back.dataset_root, index.root)
        self.assertEqual(back.n_chunks, index.n_chunks)
        self.assertEqual(back.n_bytes, os.path.getsize(path))


if __name__ == "__main__":
    unittest.main(verbosity=2)


class IndexerCommandTests(CorpusFixture):
    """The command that turns a file into a root.

    It is how a pin is defined and how anyone checks one, so the two verdicts
    it can reach — this is the corpus that network agreed on, or it is not —
    matter more than the figures it prints on the way.
    """

    def run_cli(self, *argv) -> tuple:
        import contextlib
        import io

        from rnet.__main__ import main
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(list(argv))
        return code, out.getvalue(), err.getvalue()

    def test_ItPrintsTheSameRootTheIndexerProduces(self):
        path = self.write(self.documents(120))
        expected = build_index(path, 4096)
        code, out, _ = self.run_cli("corpus-index", "--corpus", path,
                                    "--target", "4096")
        self.assertEqual(code, 0)
        self.assertIn(expected.root.hex(), out)
        self.assertIn(f"{expected.n_chunks:,}", out)

    def test_ANetworkWithNoCorpusIsSaidSoRatherThanFailed(self):
        path = self.write(self.documents(20))
        code, out, _ = self.run_cli("corpus-index", "--corpus", path,
                                    "--target", "4096", "--network", "regtest")
        self.assertEqual(code, 0)
        self.assertIn("pins no corpus", out)

    def test_AMismatchAgainstAPinnedNetworkExitsNonZero(self):
        """The whole reason the command exists: finding out after a two-day
        build that the file is not what the network agreed on, without having
        to start a daemon and read an exception."""
        path = self.write(self.documents(20))
        code, _, err = self.run_cli("corpus-index", "--corpus", path,
                                    "--target", "4096", "--network", "main")
        self.assertEqual(code, 1)
        self.assertIn("DIFFERENT from main", err)

    def test_TheIndexIsCachedAndReused(self):
        path = self.write(self.documents(60))
        self.run_cli("corpus-index", "--corpus", path, "--target", "4096")
        self.assertTrue(os.path.exists(path + ".rnidx"))
        code, out, _ = self.run_cli("corpus-index", "--corpus", path,
                                    "--target", "4096")
        self.assertEqual(code, 0)
        self.assertIn("cached", out)

    def test_RebuildIgnoresTheCache(self):
        path = self.write(self.documents(60))
        self.run_cli("corpus-index", "--corpus", path, "--target", "4096")
        code, out, _ = self.run_cli("corpus-index", "--corpus", path,
                                    "--target", "4096", "--rebuild")
        self.assertEqual(code, 0)
        self.assertNotIn("cached", out)

    def test_AMissingFileIsAMessageNotATraceback(self):
        code, _, err = self.run_cli("corpus-index",
                                    "--corpus", os.path.join(self.dir, "absent"))
        self.assertEqual(code, 1)
        self.assertTrue(err.startswith("rnet:"), err)
