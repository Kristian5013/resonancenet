#pragma once

#include <cstdint>
#include <string>

#include "core/types.h"
#include "primitives/block_header.h"

namespace rnet::chain {

/// CBlockIndex — in-memory block tree node.
/// Each block in the block tree has one CBlockIndex.
/// Linked via prev pointer to form the chain.
class CBlockIndex {
public:
    /// Block hash (cached on first computation)
    rnet::uint256 block_hash;

    /// Pointer to the previous block index, or nullptr for genesis
    CBlockIndex* prev = nullptr;

    /// Height in the chain (0 for genesis)
    int height = 0;

    /// Full block header
    primitives::CBlockHeader header;

    // --- PoT fields cached from header for fast access ---
    float val_loss = 0.0f;
    uint32_t d_model = 384;
    uint32_t n_layers = 6;

    // --- Disk position ---
    int file_number = -1;    ///< Block file number
    int64_t data_pos = -1;   ///< Byte offset in block file

    // --- Validation status ---
    enum Status : uint32_t {
        HEADER_VALID    = 0,
        TREE_VALID      = 1,
        SCRIPTS_VALID   = 2,
        FULLY_VALIDATED = 3,
    };
    Status status = HEADER_VALID;

    // --- Chain work / cumulative stats ---
    int64_t chain_tx = 0;         ///< Total txs in chain up to this block
    uint64_t timestamp = 0;       ///< Cached header timestamp

    CBlockIndex() = default;

    /// Construct from a block header
    explicit CBlockIndex(const primitives::CBlockHeader& hdr);

    /// Get the block hash (computes and caches if needed)
    const rnet::uint256& get_block_hash() const;

    /// Get the ancestor at a given height
    CBlockIndex* get_ancestor(int target_height);
    const CBlockIndex* get_ancestor(int target_height) const;

    /// Build a block locator from this index (exponentially-spaced hashes)
    std::vector<rnet::uint256> get_locator() const;

    /// Check if this block is an ancestor of another
    bool is_ancestor_of(const CBlockIndex* other) const;

    /// Human-readable
    std::string to_string() const;

    bool operator==(const CBlockIndex& other) const {
        return block_hash == other.block_hash;
    }
};

}  // namespace rnet::chain
