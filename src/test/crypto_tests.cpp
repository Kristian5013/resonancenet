#include "crypto/merkle.h"
#include "crypto/sha3.h"
#include "test/framework.h"
#include "util/hex.h"

using namespace rnet;
using rnet::crypto::Hash256;

// FIPS-202 known-answer vectors. If these ever change, the vendored primitive is
// wrong and every hash in the protocol is wrong with it.
RNET_TEST(Sha3, KnownAnswers) {
    RNET_CHECK_STREQ(util::ToHex(crypto::Sha3_256(std::string_view(""))),
                     "a7ffc6f8bf1ed76651c14756a061d662f580ff4de43b49fa82d80a4b80f8434a");
    RNET_CHECK_STREQ(util::ToHex(crypto::Sha3_256(std::string_view("abc"))),
                     "3a985da74fe225b2045c172d6bd390bd855f086e3e9d525b46bfe24511431532");
    RNET_CHECK_STREQ(
        util::ToHex(crypto::Sha3_256(std::string_view("The quick brown fox jumps over the lazy dog"))),
        "69070dda01975c8c120c3aada1b282394e7f032fa9cf32f4cb2259a0897dfc04");
}

RNET_TEST(Sha3, IncrementalMatchesOneShot) {
    const std::string part1 = "ResonanceNet ";
    const std::string part2 = "deterministic training";
    crypto::Sha3Hasher h;
    h.Update(part1).Update(part2);
    RNET_CHECK_EQ(h.Finalize(), crypto::Sha3_256(std::string_view(part1 + part2)));
}

// Boundary lengths around the 136-byte SHA3-256 rate: absorption bugs hide here.
RNET_TEST(Sha3, RateBoundaries) {
    for (size_t len : {size_t{0}, size_t{1}, size_t{135}, size_t{136}, size_t{137}, size_t{272}}) {
        std::vector<uint8_t> data(len, 0xAB);
        crypto::Sha3Hasher h;
        h.Update(data);
        RNET_CHECK_EQ(h.Finalize(), crypto::Sha3_256(data));
    }
}

RNET_TEST(Sha3, DoubleIsHashOfHash) {
    const std::string msg = "Lattica";
    const Hash256 once = crypto::Sha3_256(std::string_view(msg));
    RNET_CHECK_EQ(crypto::Sha3_256d(std::span<const uint8_t>(
                      reinterpret_cast<const uint8_t*>(msg.data()), msg.size())),
                  crypto::Sha3_256(std::span<const uint8_t>(once.data(), once.size())));
}

// Domain separation is what makes an odd promoted node unambiguous; without it a
// leaf could be substituted for an internal node.
RNET_TEST(Merkle, LeafAndNodeAreDomainSeparated) {
    const std::vector<uint8_t> data(64, 0x11);
    const Hash256 leaf = crypto::MerkleLeaf(data);
    Hash256 half{};
    std::copy(data.begin(), data.begin() + 32, half.begin());
    RNET_CHECK(leaf != crypto::MerkleNode(half, half));
}

RNET_TEST(Merkle, SingleLeafRootIsThatLeaf) {
    const std::vector<uint8_t> data{1, 2, 3};
    const Hash256 leaf = crypto::MerkleLeaf(data);
    RNET_CHECK_EQ(crypto::MerkleRoot({leaf}), leaf);
}

// The CVE-2012-2459 shape: a tree whose last node is promoted must NOT collide
// with a tree that genuinely contains that node twice.
RNET_TEST(Merkle, PromotedOddNodeDoesNotCollideWithDuplicate) {
    std::vector<Hash256> three;
    for (uint8_t i = 0; i < 3; ++i) three.push_back(crypto::MerkleLeaf(std::vector<uint8_t>{i}));

    std::vector<Hash256> four = three;
    four.push_back(three[2]);   // duplicate the last leaf

    RNET_CHECK(crypto::MerkleRoot(three) != crypto::MerkleRoot(four));
}

RNET_TEST(Merkle, ProofsVerifyForEveryLeaf) {
    for (size_t n : {size_t{1}, size_t{2}, size_t{3}, size_t{5}, size_t{8}, size_t{9}, size_t{17}}) {
        std::vector<Hash256> leaves;
        for (size_t i = 0; i < n; ++i) {
            leaves.push_back(crypto::MerkleLeaf(std::vector<uint8_t>{static_cast<uint8_t>(i), 0x5A}));
        }
        const Hash256 root = crypto::MerkleRoot(leaves);
        for (size_t i = 0; i < n; ++i) {
            auto proof = crypto::BuildMerkleProof(leaves, i);
            RNET_CHECK_OK(proof);
            if (!crypto::VerifyMerkleProof(leaves[i], proof.value(), root)) {
                RNET_FAIL("proof failed to verify for leaf {} of {}", i, n);
            }
        }
    }
}

RNET_TEST(Merkle, TamperedLeafFailsVerification) {
    std::vector<Hash256> leaves;
    for (uint8_t i = 0; i < 6; ++i) leaves.push_back(crypto::MerkleLeaf(std::vector<uint8_t>{i}));
    const Hash256 root = crypto::MerkleRoot(leaves);

    auto proof = crypto::BuildMerkleProof(leaves, 2);
    RNET_CHECK_OK(proof);

    const Hash256 forged = crypto::MerkleLeaf(std::vector<uint8_t>{0xFF});
    RNET_CHECK(!crypto::VerifyMerkleProof(forged, proof.value(), root));
}

RNET_TEST(Merkle, TruncatedOrPaddedProofIsRejected) {
    std::vector<Hash256> leaves;
    for (uint8_t i = 0; i < 8; ++i) leaves.push_back(crypto::MerkleLeaf(std::vector<uint8_t>{i}));
    const Hash256 root = crypto::MerkleRoot(leaves);

    auto proof = crypto::BuildMerkleProof(leaves, 3);
    RNET_CHECK_OK(proof);

    auto shortened = proof.value();
    shortened.path.pop_back();
    RNET_CHECK(!crypto::VerifyMerkleProof(leaves[3], shortened, root));

    auto padded = proof.value();
    padded.path.push_back(root);
    RNET_CHECK(!crypto::VerifyMerkleProof(leaves[3], padded, root));
}
