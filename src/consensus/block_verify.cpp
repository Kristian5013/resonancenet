#include "consensus/block_verify.h"
#include "consensus/merkle.h"
#include "consensus/proof_of_training.h"
#include "consensus/tx_verify.h"

#include <set>

namespace rnet::consensus {

bool check_block(const primitives::CBlock& block,
                 ValidationState& state,
                 const ConsensusParams& params) {
    // Must have at least one transaction
    if (block.vtx.empty()) {
        state.invalid("bad-blk-length");
        return false;
    }

    // First transaction must be coinbase
    if (!block.vtx[0]->is_coinbase()) {
        state.invalid("bad-cb-missing");
        return false;
    }

    // Only the first transaction may be coinbase
    for (size_t i = 1; i < block.vtx.size(); ++i) {
        if (block.vtx[i]->is_coinbase()) {
            state.invalid("bad-cb-multiple");
            return false;
        }
    }

    // Check merkle root
    rnet::uint256 computed_root = block_merkle_root(block);
    if (!(computed_root == block.merkle_root)) {
        state.invalid("bad-txnmrklroot");
        return false;
    }

    // Check for duplicate txids
    std::set<rnet::uint256> seen_txids;
    for (const auto& tx : block.vtx) {
        if (!seen_txids.insert(tx->txid()).second) {
            state.invalid("bad-txns-duplicate");
            return false;
        }
    }

    // Block size limit
    size_t block_size = block.get_block_size();
    if (block_size > params.max_block_size) {
        state.invalid("bad-blk-size");
        return false;
    }

    // Sigop count limit
    int total_sigops = 0;
    for (const auto& tx : block.vtx) {
        // Count sigops from output scripts
        for (const auto& txout : tx->vout()) {
            // Each OP_CHECKSIG / OP_CHECKSIGVERIFY counts as 1
            // Each OP_CHECKMULTISIG / OP_CHECKMULTISIGVERIFY counts as 20
            // Simple heuristic: count 0xAC (OP_CHECKSIG) occurrences
            for (uint8_t byte : txout.script_pub_key) {
                if (byte == 0xAC) total_sigops += 1;
            }
        }
    }
    if (total_sigops > params.max_block_sigops) {
        state.invalid("bad-blk-sigops");
        return false;
    }

    // Validate each transaction
    for (const auto& tx : block.vtx) {
        if (!check_transaction(*tx, state)) {
            return false;
        }
    }

    return true;
}

bool check_block_header(const primitives::CBlockHeader& header,
                        const primitives::CBlockHeader& parent,
                        ValidationState& state,
                        const ConsensusParams& params) {
    // Genesis block: only basic sanity
    if (header.is_genesis()) {
        if (header.height != 0) {
            state.invalid("bad-genesis-height");
            return false;
        }
        return true;
    }

    // Height must be parent + 1
    if (header.height != parent.height + 1) {
        state.invalid("bad-height");
        return false;
    }

    // prev_hash must match parent's hash
    if (!(header.prev_hash == parent.hash())) {
        state.invalid("bad-prevblk");
        return false;
    }

    // Timestamp must be strictly greater than parent
    if (header.timestamp <= parent.timestamp) {
        state.invalid("bad-timestamp");
        return false;
    }

    // Verify Proof-of-Training header fields
    if (!verify_pot_header(header, parent, params, state)) {
        return false;  // state already set by verify_pot_header
    }

    return true;
}

}  // namespace rnet::consensus
