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

/// Block template: a partially-assembled block ready for PoT solving.
/// Contains the header fields (except checkpoint_hash, val_loss, signature)
/// and the full transaction list including coinbase.
struct BlockTemplate {
    /// The assembled block (header + txs). The miner fills in:
    ///   - checkpoint_hash (after training)
    ///   - val_loss (after evaluation)
    ///   - signature (after signing)
    primitives::CBlock block;

    /// Block reward breakdown.
    consensus::BlockReward reward;

    /// Growth result for this block.
    consensus::GrowthResult growth;

    /// Total fees collected from included transactions.
    int64_t total_fees = 0;

    /// Number of transactions (excluding coinbase).
    size_t tx_count() const {
        return block.vtx.empty() ? 0 : block.vtx.size() - 1;
    }
};

/// Create a block template from the parent header and available transactions.
///
/// @param parent_header  The tip block's header.
/// @param txs            Candidate transactions (sorted by fee rate, highest first).
/// @param tx_fees        Fee for each transaction (parallel to txs vector).
/// @param miner_pubkey   Miner's Ed25519 public key.
/// @param emission       Current emission state for reward calculation.
/// @param params         Consensus parameters.
/// @return               A block template ready for PoT solving.
BlockTemplate create_block_template(
    const primitives::CBlockHeader& parent_header,
    const std::vector<primitives::CTransactionRef>& txs,
    const std::vector<int64_t>& tx_fees,
    const crypto::Ed25519PublicKey& miner_pubkey,
    const consensus::EmissionState& emission,
    const consensus::ConsensusParams& params);

}  // namespace rnet::miner
