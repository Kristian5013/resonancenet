#include "crypto/merkle.h"

namespace rnet::crypto {

namespace {
constexpr uint8_t kLeafPrefix = 0x00;
constexpr uint8_t kNodePrefix = 0x01;
}  // namespace

Hash256 MerkleLeaf(std::span<const uint8_t> data) {
    Sha3Hasher h;
    const uint8_t prefix = kLeafPrefix;
    h.Update(std::span<const uint8_t>(&prefix, 1));
    h.Update(data);
    return h.Finalize();
}

Hash256 MerkleNode(const Hash256& left, const Hash256& right) {
    Sha3Hasher h;
    const uint8_t prefix = kNodePrefix;
    h.Update(std::span<const uint8_t>(&prefix, 1));
    h.Update(std::span<const uint8_t>(left.data(), left.size()));
    h.Update(std::span<const uint8_t>(right.data(), right.size()));
    return h.Finalize();
}

Hash256 MerkleRoot(std::vector<Hash256> level) {
    if (level.empty()) return MerkleLeaf({});
    while (level.size() > 1) {
        std::vector<Hash256> next;
        next.reserve((level.size() + 1) / 2);
        for (size_t i = 0; i + 1 < level.size(); i += 2) {
            next.push_back(MerkleNode(level[i], level[i + 1]));
        }
        if (level.size() % 2 == 1) {
            // Odd node is promoted unchanged — unambiguous thanks to the leaf/node
            // domain separation, unlike Bitcoin's duplicate-the-last-node padding.
            next.push_back(level.back());
        }
        level = std::move(next);
    }
    return level.front();
}

Result<MerkleProof> BuildMerkleProof(const std::vector<Hash256>& leaves, uint64_t index) {
    if (leaves.empty()) return Err("merkle: empty tree has no proofs");
    if (index >= leaves.size()) return Err("merkle: index out of range");

    MerkleProof proof;
    proof.index = index;
    proof.leaf_count = leaves.size();

    std::vector<Hash256> level = leaves;
    uint64_t pos = index;
    while (level.size() > 1) {
        const bool has_sibling = (pos % 2 == 1) || (pos + 1 < level.size());
        if (has_sibling) {
            const uint64_t sibling = (pos % 2 == 0) ? pos + 1 : pos - 1;
            proof.path.push_back(level[sibling]);
        }
        // Promoted odd nodes contribute no sibling and keep their relative position.

        std::vector<Hash256> next;
        next.reserve((level.size() + 1) / 2);
        for (size_t i = 0; i + 1 < level.size(); i += 2) {
            next.push_back(MerkleNode(level[i], level[i + 1]));
        }
        if (level.size() % 2 == 1) next.push_back(level.back());
        pos /= 2;
        level = std::move(next);
    }
    return proof;
}

bool VerifyMerkleProof(const Hash256& leaf, const MerkleProof& proof, const Hash256& expected_root) {
    if (proof.leaf_count == 0 || proof.index >= proof.leaf_count) return false;

    Hash256 acc = leaf;
    uint64_t pos = proof.index;
    uint64_t width = proof.leaf_count;
    size_t used = 0;

    while (width > 1) {
        const bool has_sibling = (pos % 2 == 1) || (pos + 1 < width);
        if (has_sibling) {
            if (used >= proof.path.size()) return false;   // proof too short
            const Hash256& sib = proof.path[used++];
            acc = (pos % 2 == 0) ? MerkleNode(acc, sib) : MerkleNode(sib, acc);
        }
        pos /= 2;
        width = (width + 1) / 2;
    }
    if (used != proof.path.size()) return false;           // proof too long
    return acc == expected_root;
}

}  // namespace rnet::crypto
