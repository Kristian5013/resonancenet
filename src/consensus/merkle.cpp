#include "consensus/merkle.h"

#include "crypto/merkle.h"

namespace rnet::consensus {

rnet::uint256 block_merkle_root(const primitives::CBlock& block) {
    if (block.vtx.empty()) {
        return rnet::uint256{};
    }

    std::vector<rnet::uint256> leaves;
    leaves.reserve(block.vtx.size());
    for (const auto& tx : block.vtx) {
        leaves.push_back(tx->txid());
    }

    return crypto::compute_merkle_root(std::move(leaves));
}

rnet::uint256 block_witness_merkle_root(const primitives::CBlock& block) {
    if (block.vtx.empty()) {
        return rnet::uint256{};
    }

    std::vector<rnet::uint256> leaves;
    leaves.reserve(block.vtx.size());

    // Coinbase wtxid is replaced with zero hash
    leaves.emplace_back();  // zero-initialized uint256

    for (size_t i = 1; i < block.vtx.size(); ++i) {
        leaves.push_back(block.vtx[i]->wtxid());
    }

    return crypto::compute_merkle_root(std::move(leaves));
}

}  // namespace rnet::consensus
