#pragma once

#include <cstdint>
#include <vector>

#include "consensus/block_reward.h"
#include "consensus/growth_policy.h"
#include "consensus/params.h"
#include "crypto/ed25519.h"
#include "primitives/block.h"
#include "primitives/transaction.h"

namespace rnet::miner {

/// Transaction entry with fee metadata for selection.
struct TxEntry {
    primitives::CTransactionRef tx;
    int64_t fee = 0;           ///< Total fee in resonances
    size_t weight = 0;         ///< Transaction weight units
    int64_t fee_rate = 0;      ///< Fee per 1000 weight units

    /// Compare by fee rate (descending) for priority ordering.
    bool operator>(const TxEntry& other) const {
        return fee_rate > other.fee_rate;
    }
};

/// Block assembler: selects transactions and builds the full block.
class BlockAssembler {
public:
    explicit BlockAssembler(const consensus::ConsensusParams& params);

    /// Add candidate transactions for block inclusion.
    /// Transactions are sorted by fee rate internally.
    void add_transactions(const std::vector<TxEntry>& entries);

    /// Assemble a block: select transactions up to weight limit,
    /// create coinbase with the given reward, and compute merkle root.
    ///
    /// @param parent       Parent block header (for height, prev_hash, etc.)
    /// @param miner_pubkey Miner's public key for coinbase output.
    /// @param reward       Block reward (base + bonus + recovered).
    /// @param growth       Growth result for header fields.
    /// @return             Assembled block with all transactions.
    primitives::CBlock assemble(
        const primitives::CBlockHeader& parent,
        const crypto::Ed25519PublicKey& miner_pubkey,
        const consensus::BlockReward& reward,
        const consensus::GrowthResult& growth);

    /// Get the total fees from selected transactions.
    int64_t total_fees() const { return total_fees_; }

    /// Get the number of selected transactions.
    size_t selected_count() const { return selected_.size(); }

private:
    const consensus::ConsensusParams& params_;
    std::vector<TxEntry> candidates_;
    std::vector<primitives::CTransactionRef> selected_;
    int64_t total_fees_ = 0;
    size_t total_weight_ = 0;

    /// Select transactions greedily by fee rate up to weight limit.
    void select_transactions();
};

}  // namespace rnet::miner
