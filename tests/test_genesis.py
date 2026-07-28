"""Tests for the trust bootstrap.

The property under test: a node believes an artifact if and only if it hashes
to a compiled-in anchor, and everything downstream of that belief is derived
from bytes that cleared the check.
"""

import hashlib
import os
import shutil
import tempfile
import unittest

from rnet.consensus import genesis
from rnet.consensus.params import NETWORKS


class GenesisTests(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="rnet-genesis-")
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)

    # -- the build is self-consistent --------------------------------------

    def test_TheTablesAgreeWithTheAnchors(self):
        """A consensus value edited without regenerating its anchor is caught
        here rather than at the first handshake with a stranger."""
        genesis.verify_build()
        for network in genesis.networks():
            genesis.verify_build(network)

    def test_EveryNetworkHasBothAnchors(self):
        self.assertEqual(sorted(NETWORKS), sorted(genesis.GENESIS_HASH))
        self.assertEqual(sorted(NETWORKS), sorted(genesis.POLICY_HASH))

    def test_EveryAnchorIsDistinct(self):
        """Two networks sharing an anchor would accept each other's artifacts."""
        anchors = list(genesis.GENESIS_HASH.values()) + list(genesis.POLICY_HASH.values())
        self.assertEqual(len(set(anchors)), len(anchors))

    def test_EveryNetworkHasADistinctMagic(self):
        magics = [r.network_magic for r, _ in NETWORKS.values()]
        self.assertEqual(len(set(magics)), len(magics))
        for name, (r, p) in NETWORKS.items():
            self.assertEqual(r.network_magic, p.network_magic, name)

    def test_AChangedValueBreaksItsAnchor(self):
        """The check has teeth: prove it fails when it should."""
        real = genesis.GENESIS_HASH["regtest"]
        genesis.GENESIS_HASH["regtest"] = "00" * 32
        try:
            with self.assertRaises(genesis.GenesisError):
                genesis.verify_build("regtest")
        finally:
            genesis.GENESIS_HASH["regtest"] = real
        genesis.verify_build("regtest")

    # -- artifacts on disk --------------------------------------------------

    def test_EveryNetworkEmitsAndLoadsBack(self):
        for network in genesis.networks():
            with self.subTest(network):
                rnet_path, rnpol_path = genesis.emit(network, self.dir)
                r = genesis.load_round(rnet_path, network)
                p = genesis.load_policy(rnpol_path, network)
                self.assertEqual(r, NETWORKS[network][0])
                self.assertEqual(p, NETWORKS[network][1])

    def test_AnEmittedArtifactHashesToItsAnchor(self):
        for network in genesis.networks():
            with self.subTest(network):
                rnet_path, rnpol_path = genesis.emit(network, self.dir)
                for path, anchors in ((rnet_path, genesis.GENESIS_HASH),
                                      (rnpol_path, genesis.POLICY_HASH)):
                    with open(path, "rb") as f:
                        self.assertEqual(hashlib.sha3_256(f.read()).hexdigest(),
                                         anchors[network])

    def test_EveryMutationOfAnArtifactIsRefused(self):
        """One flipped bit anywhere, and the file is somebody else's."""
        rnet_path, _ = genesis.emit("regtest", self.dir)
        with open(rnet_path, "rb") as f:
            original = f.read()
        for i in range(len(original)):
            mutated = bytearray(original)
            mutated[i] ^= 0x01
            with open(rnet_path, "wb") as f:
                f.write(bytes(mutated))
            with self.assertRaises(genesis.GenesisError, msg=f"byte {i}"):
                genesis.load_round(rnet_path, "regtest")

    def test_AnArtifactFromAnotherNetworkIsRefused(self):
        """The check that stops a node joining the wrong network by filename."""
        genesis.emit("main", self.dir)
        genesis.emit("test", self.dir)
        with self.assertRaises(genesis.GenesisError):
            genesis.load_round(os.path.join(self.dir, "main.rnet"), "test")
        with self.assertRaises(genesis.GenesisError):
            genesis.load_policy(os.path.join(self.dir, "test.rnpol"), "main")

    def test_APolicyArtifactIsNotARoundArtifact(self):
        rnet_path, rnpol_path = genesis.emit("regtest", self.dir)
        with self.assertRaises(genesis.GenesisError):
            genesis.load_round(rnpol_path, "regtest")
        with self.assertRaises(genesis.GenesisError):
            genesis.load_policy(rnet_path, "regtest")

    def test_AnUnknownNetworkIsRefused(self):
        """And says which networks exist, since the usual cause is a typo."""
        with self.assertRaises(genesis.GenesisError):
            genesis.load_round("/nonexistent", "mainnet")
        for call in (lambda: genesis.round_descriptor("mainnet"),
                     lambda: genesis.policy_descriptor("mainnet"),
                     lambda: genesis.verify_build("mainnet")):
            with self.assertRaises(genesis.GenesisError) as ctx:
                call()
            self.assertIn("regtest", str(ctx.exception))

    def test_ANetworkWithNoAnchorCannotShip(self):
        """Otherwise the build emits artifacts it would then refuse to load."""
        real = genesis.GENESIS_HASH.pop("regtest")
        try:
            with self.assertRaises(genesis.GenesisError) as ctx:
                genesis.verify_build("regtest")
            self.assertIn("no anchor", str(ctx.exception))
        finally:
            genesis.GENESIS_HASH["regtest"] = real

    def test_EmittingIsAtomic(self):
        """No .tmp left behind, so a crash mid-write cannot leave a half file
        that anchors to nothing and looks like corruption."""
        genesis.emit("regtest", self.dir)
        self.assertEqual([f for f in os.listdir(self.dir) if f.endswith(".tmp")], [])

    # -- what the networks actually say ------------------------------------

    def test_MainTrainsTheModelThatActuallyTrained(self):
        r = genesis.round_descriptor("main")
        self.assertEqual(r.model.parameter_count(), 397_728_768)
        self.assertFalse(r.model.is_moe)

    def test_MainCarriesTheCorpusTheTreeDeclares(self):
        """A wiring test, and it used to be a copy of the chunk count.

        The copy went stale the first time the network was re-pinned, which is
        the wrong failure: it says nothing about whether the pin is right, only
        that somebody changed it. What is worth checking is that the descriptor
        carries what params.py declares rather than dropping it — a
        `dataset_chunks` silently left at zero would mean a round pinning a
        corpus whose size nothing agreed on.

        Whether the pin names the RIGHT corpus is not a question a test can
        answer. `rnet corpus-index --network main` answers it, against a file.
        """
        from rnet.consensus import params
        r = genesis.round_descriptor("main")
        self.assertEqual(r.dataset_root, params.CORPUS_ROOT)
        self.assertEqual(r.dataset_chunks, params.CORPUS_CHUNKS)
        self.assertNotEqual(r.dataset_root, bytes(32))
        self.assertGreater(r.dataset_chunks, 0)
        # The pin names a snapshot. One that does not is one nobody can rebuild.
        self.assertEqual(len(params.CORPUS_REVISION), 40, params.CORPUS_REVISION)

    def test_TheMixtureNetworkIsDescribedAndFitsAShard(self):
        r = genesis.round_descriptor("moe")
        self.assertTrue(r.model.is_moe)
        self.assertEqual(r.model.parameter_count(), 29_408_635_904)
        self.assertLess(r.model.bytes_per_shard(16) / 2**30, 16.0)

    def test_MainAndTheMixtureShareACorpus(self):
        """The corpus is derived from bytes, so it survives a model change."""
        main = genesis.round_descriptor("main")
        moe = genesis.round_descriptor("moe")
        self.assertEqual(main.dataset_root, moe.dataset_root)
        self.assertEqual(main.tokenizer_hash, moe.tokenizer_hash)
        self.assertNotEqual(main.id, moe.id)

    def test_RegtestNeedsNoCorpusAndNoFusedKernel(self):
        """It has to run anywhere, including where flash attention does not."""
        r = genesis.round_descriptor("regtest")
        self.assertEqual(r.dataset_root, bytes(32))
        self.assertEqual(r.dataset_chunks, 0)
        self.assertTrue(r.byte_level_tokenizer)
        self.assertNotEqual(r.determinism_class,
                            genesis.round_descriptor("main").determinism_class)

    def test_EveryNetworkStartsInShadowMode(self):
        """Nothing is punished until the verification path has been watched."""
        for network in genesis.networks():
            self.assertTrue(genesis.policy_descriptor(network).shadow_mode, network)

    def test_DescribeMentionsEveryAnchor(self):
        for network in genesis.networks():
            text = genesis.describe(network)
            self.assertIn(genesis.GENESIS_HASH[network], text)
            self.assertIn(genesis.POLICY_HASH[network], text)
            self.assertIn(genesis.WEIGHTS_HASH[network], text)

    # -- the fourth anchor --------------------------------------------------

    def test_EveryNetworkHasAWeightsAnchor(self):
        self.assertEqual(sorted(NETWORKS), sorted(genesis.WEIGHTS_HASH))
        self.assertEqual(len(set(genesis.WEIGHTS_HASH.values())), len(NETWORKS))

    def test_TheRegtestWeightsDeriveToTheirAnchor(self):
        """Checked here because regtest is 4 million parameters and instant.

        The other three are real work — 1.5 seconds for the dense 400M and 85
        for the 29.4 billion mixture — so they are checked by
        `rnet genesis-weights`, deliberately and not on every test run.
        """
        self.assertEqual(genesis.verify_weights("regtest").hex(),
                         genesis.WEIGHTS_HASH["regtest"])

    def test_AChangedWeightsAnchorIsCaught(self):
        real = genesis.WEIGHTS_HASH["regtest"]
        genesis.WEIGHTS_HASH["regtest"] = "00" * 32
        try:
            with self.assertRaises(genesis.GenesisError) as ctx:
                genesis.verify_weights("regtest")
            self.assertIn("consensus change", str(ctx.exception))
        finally:
            genesis.WEIGHTS_HASH["regtest"] = real

    def test_TheWeightsAnchorDependsOnTheGenesisHash(self):
        """Which is what makes it the fourth anchor and not a fifth constant."""
        from rnet.consensus.init import weights_hash
        spec = genesis.round_descriptor("regtest").model
        theirs = weights_hash(spec, bytes.fromhex(genesis.GENESIS_HASH["main"]))
        self.assertNotEqual(theirs.hex(), genesis.WEIGHTS_HASH["regtest"])


class CommandLineTests(unittest.TestCase):
    """The auditor's path: no build, no GPU, standard library only."""

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="rnet-cli-")
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)

    def test_EmitThenVerifyRoundTrips(self):
        from rnet.__main__ import main
        self.assertEqual(main(["genesis-emit", "--out", self.dir]), 0)
        self.assertEqual(main(["genesis-verify", "--dir", self.dir]), 0)

    def test_VerifyFailsOnATamperedArtifact(self):
        from rnet.__main__ import main
        main(["genesis-emit", "regtest", "--out", self.dir])
        path = os.path.join(self.dir, "regtest.rnet")
        with open(path, "r+b") as f:
            f.seek(20)
            f.write(b"\xff")
        self.assertEqual(main(["genesis-verify", "regtest", "--dir", self.dir]), 1)

    def test_ShowWorksForEveryNetwork(self):
        from rnet.__main__ import main
        self.assertEqual(main(["genesis-show"]), 0)

    def test_AnUnknownNetworkExitsNonZero(self):
        from rnet.__main__ import main
        self.assertEqual(main(["genesis-show", "mainnet"]), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
