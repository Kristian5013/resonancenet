#include "primitives/block.h"

#include "crypto/keccak.h"

namespace rnet::primitives {

rnet::uint256 CBlock::compute_merkle_root() const {
    if (vtx.empty()) {
        return rnet::uint256{};
    }

    // Collect transaction ids
    std::vector<rnet::uint256> leaves;
    leaves.reserve(vtx.size());
    for (const auto& tx : vtx) {
        leaves.push_back(tx->txid());
    }

    // Build merkle tree iteratively
    while (leaves.size() > 1) {
        // If odd number, duplicate the last element
        if (leaves.size() % 2 != 0) {
            leaves.push_back(leaves.back());
        }

        std::vector<rnet::uint256> next_level;
        next_level.reserve(leaves.size() / 2);

        for (size_t i = 0; i < leaves.size(); i += 2) {
            // Concatenate two hashes and compute keccak256d
            std::vector<uint8_t> combined(64);
            std::memcpy(combined.data(), leaves[i].data(), 32);
            std::memcpy(combined.data() + 32, leaves[i + 1].data(), 32);
            next_level.push_back(
                crypto::keccak256d(std::span<const uint8_t>(combined)));
        }

        leaves = std::move(next_level);
    }

    return leaves[0];
}

size_t CBlock::get_block_size() const {
    size_t total = 0;
    for (const auto& tx : vtx) {
        total += tx->get_total_size();
    }
    return total;
}

size_t CBlock::get_block_weight() const {
    size_t total = 0;
    for (const auto& tx : vtx) {
        total += tx->get_weight();
    }
    return total;
}

std::string CBlock::to_string() const {
    std::string result = "CBlock(";
    result += CBlockHeader::to_string();
    result += ", txs=" + std::to_string(vtx.size());
    result += ")";
    return result;
}

}  // namespace rnet::primitives
