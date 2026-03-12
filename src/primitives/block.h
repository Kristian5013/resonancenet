#pragma once

#include <string>
#include <vector>

#include "core/serialize.h"
#include "primitives/block_header.h"
#include "primitives/transaction.h"

namespace rnet::primitives {

/// CBlock — a full block: header + transactions.
struct CBlock : public CBlockHeader {
    std::vector<CTransactionRef> vtx;

    CBlock() = default;

    /// Construct with just the header
    explicit CBlock(const CBlockHeader& header)
        : CBlockHeader(header) {}

    /// Clear the block (header + transactions)
    void set_null() {
        *static_cast<CBlockHeader*>(this) = CBlockHeader{};
        vtx.clear();
    }

    /// Compute the merkle root from the transactions.
    /// Does NOT update merkle_root; call this and set it explicitly.
    rnet::uint256 compute_merkle_root() const;

    /// Get the coinbase transaction (first tx), or nullptr.
    CTransactionRef get_coinbase() const {
        if (!vtx.empty()) return vtx[0];
        return nullptr;
    }

    /// Total serialized size of all transactions (with witness).
    size_t get_block_size() const;

    /// Total weight of all transactions.
    size_t get_block_weight() const;

    /// Number of transactions.
    size_t tx_count() const { return vtx.size(); }

    /// Human-readable
    std::string to_string() const;

    /// Serialization: header + transaction vector
    template<typename Stream>
    void serialize(Stream& s) const {
        // Serialize header fields via parent
        CBlockHeader::serialize(s);

        // Serialize transaction count + transactions
        core::serialize_compact_size(s, vtx.size());
        for (const auto& tx : vtx) {
            tx->serialize(s);
        }
    }

    template<typename Stream>
    void unserialize(Stream& s) {
        // Unserialize header fields via parent
        CBlockHeader::unserialize(s);

        // Unserialize transactions
        uint64_t count = core::unserialize_compact_size(s);
        if (count > 0x02000000) {
            return;  // Sanity limit
        }
        vtx.resize(static_cast<size_t>(count));
        for (auto& tx_ref : vtx) {
            auto mtx_ptr = std::make_shared<CTransaction>();
            auto& mtx = const_cast<CTransaction&>(*mtx_ptr);
            mtx.unserialize(s);
            tx_ref = mtx_ptr;
        }
    }
};

/// CBlockLocator — a compact representation of the chain for locating
/// the fork point with a peer. Contains a list of block hashes at
/// exponentially-spaced heights.
struct CBlockLocator {
    std::vector<rnet::uint256> have;

    CBlockLocator() = default;
    explicit CBlockLocator(std::vector<rnet::uint256> hashes)
        : have(std::move(hashes)) {}

    bool is_null() const { return have.empty(); }
    void set_null() { have.clear(); }

    SERIALIZE_METHODS(
        READWRITE(self.have);
    )
};

}  // namespace rnet::primitives
