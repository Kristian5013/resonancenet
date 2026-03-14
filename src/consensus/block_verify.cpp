// Copyright (c) 2024-2026 The ResonanceNet Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "consensus/block_verify.h"

#include "consensus/merkle.h"
#include "consensus/proof_of_training.h"
#include "consensus/tx_verify.h"
#include "core/time.h"
#include "crypto/ed25519.h"

#include <cstddef>
#include <cstdint>
#include <set>
#include <span>

namespace rnet::consensus {

// ---------------------------------------------------------------------------
// count_script_sigops
// ---------------------------------------------------------------------------
// Counts signature operations in a script by walking opcodes.
//
// The function steps through raw script bytes, skipping over pushed data
// payloads so that embedded byte values are never mistaken for opcodes.
//
// Opcode categories:
//   0x01..0x4B  direct push (skip 1 + N bytes)
//   0x4C        OP_PUSHDATA1 (skip 2 + len bytes)
//   0x4D        OP_PUSHDATA2 (skip 3 + len_le16 bytes)
//   0xAC/0xAD   OP_CHECKSIG / OP_CHECKSIGVERIFY   (+1 sigop)
//   0xAE/0xAF   OP_CHECKMULTISIG / OP_CHECKMULTISIGVERIFY (+20 sigops,
//               worst-case assumption because key count is unknown here)
//
// Returns the total sigop count for the script.
// ---------------------------------------------------------------------------
static int count_script_sigops(const std::vector<uint8_t>& script)
{
    int sigops = 0;
    size_t i = 0;

    while (i < script.size()) {
        const uint8_t opcode = script[i];

        // 1. Direct push (1..75 bytes): opcode IS the length.
        if (opcode >= 1 && opcode <= 75) {
            i += 1 + opcode;
            continue;
        }

        // 2. OP_PUSHDATA1: next byte is the length.
        if (opcode == 0x4C) {
            if (i + 1 >= script.size()) break;
            const uint8_t len = script[i + 1];
            i += 2 + len;
            continue;
        }

        // 3. OP_PUSHDATA2: next two bytes (little-endian) are the length.
        if (opcode == 0x4D) {
            if (i + 2 >= script.size()) break;
            const uint16_t len = static_cast<uint16_t>(script[i + 1])
                               | (static_cast<uint16_t>(script[i + 2]) << 8);
            i += 3 + len;
            continue;
        }

        // 4. OP_CHECKSIG / OP_CHECKSIGVERIFY — one sigop each.
        if (opcode == 0xAC || opcode == 0xAD) {
            sigops += 1;
        }

        // 5. OP_CHECKMULTISIG / OP_CHECKMULTISIGVERIFY — 20 sigops (worst case).
        if (opcode == 0xAE || opcode == 0xAF) {
            sigops += 20;
        }

        ++i;
    }

    return sigops;
}

// ---------------------------------------------------------------------------
// check_block
// ---------------------------------------------------------------------------
// Context-free structural validation of a full block (header + transactions).
//
// Checks performed (in order):
//   1. Non-empty transaction vector
//   2. First transaction is coinbase
//   3. No additional coinbase transactions
//   4. Merkle root matches recomputed value
//   5. No duplicate txids
//   6. Serialised block size within consensus limit
//   7. Aggregate sigop count within consensus limit
//   8. Each transaction passes individual checks (check_transaction)
//
// This function does NOT verify header PoT, signatures, or UTXO state.
// Those are handled by check_block_header and contextual validation.
// ---------------------------------------------------------------------------
bool check_block(const primitives::CBlock& block,
                 ValidationState& state,
                 const ConsensusParams& params)
{
    // 1. Block must contain at least one transaction.
    if (block.vtx.empty()) {
        state.invalid("bad-blk-length");
        return false;
    }

    // 2. First transaction must be the coinbase.
    if (!block.vtx[0]->is_coinbase()) {
        state.invalid("bad-cb-missing");
        return false;
    }

    // 3. Only the first transaction may be a coinbase.
    for (size_t i = 1; i < block.vtx.size(); ++i) {
        if (block.vtx[i]->is_coinbase()) {
            state.invalid("bad-cb-multiple");
            return false;
        }
    }

    // 4. Recompute the Merkle root and compare to the header commitment.
    const rnet::uint256 computed_root = block_merkle_root(block);
    if (!(computed_root == block.merkle_root)) {
        state.invalid("bad-txnmrklroot");
        return false;
    }

    // 5. Reject duplicate txids (would make Merkle proofs ambiguous).
    std::set<rnet::uint256> seen_txids;
    for (const auto& tx : block.vtx) {
        if (!seen_txids.insert(tx->txid()).second) {
            state.invalid("bad-txns-duplicate");
            return false;
        }
    }

    // 6. Serialised block size must not exceed the consensus maximum.
    const size_t block_size = block.get_block_size();
    if (block_size > params.max_block_size) {
        state.invalid("bad-blk-size");
        return false;
    }

    // 7. Total sigops across all transaction outputs must be within limit.
    int total_sigops = 0;
    for (const auto& tx : block.vtx) {
        for (const auto& txout : tx->vout()) {
            total_sigops += count_script_sigops(txout.script_pub_key);
        }
    }
    if (total_sigops > params.max_block_sigops) {
        state.invalid("bad-blk-sigops");
        return false;
    }

    // 8. Every transaction must pass context-free checks individually.
    for (const auto& tx : block.vtx) {
        if (!check_transaction(*tx, state)) {
            return false;
        }
    }

    return true;
}

// ---------------------------------------------------------------------------
// check_block_header
// ---------------------------------------------------------------------------
// Validates a block header against its parent and the consensus parameters.
//
// For the genesis block (header.is_genesis()):
//   1. Height must be zero
//
// For all other blocks:
//   1. Height must be exactly parent.height + 1
//   2. prev_hash must equal parent.hash()
//   3. Timestamp must be strictly greater than parent timestamp
//   4. Timestamp must not be more than 7 200 seconds (2 hours) in the future
//   5. Proof-of-Training header fields must validate
//   6. Ed25519 miner signature must verify over the unsigned header bytes
// ---------------------------------------------------------------------------
bool check_block_header(const primitives::CBlockHeader& header,
                        const primitives::CBlockHeader& parent,
                        ValidationState& state,
                        const ConsensusParams& params)
{
    // --- Genesis block special case ---
    if (header.is_genesis()) {
        // 1. Genesis height must be zero.
        if (header.height != 0) {
            state.invalid("bad-genesis-height");
            return false;
        }
        return true;
    }

    // --- Non-genesis blocks ---

    // 1. Height must be exactly one more than the parent.
    if (header.height != parent.height + 1) {
        state.invalid("bad-height");
        return false;
    }

    // 2. prev_hash must chain to the parent block.
    if (!(header.prev_hash == parent.hash())) {
        state.invalid("bad-prevblk");
        return false;
    }

    // 3. Timestamp must advance beyond the parent (no ties allowed).
    if (header.timestamp <= parent.timestamp) {
        state.invalid("bad-timestamp");
        return false;
    }

    // 3b. Minimum block interval — prevents blocks arriving too fast
    //     even before the difficulty retarget can react.
    //     Default: 300 seconds (5 minutes) on mainnet.
    if (header.timestamp < parent.timestamp + static_cast<uint64_t>(params.min_block_interval)) {
        state.invalid("bad-timestamp-too-soon");
        return false;
    }

    // 4. Timestamp must not be too far in the future.
    if (header.timestamp > core::get_time() + 7'200) { // seconds (2 hours)
        state.invalid("bad-timestamp-future");
        return false;
    }

    // 5. Proof-of-Training commitment must validate.
    if (!verify_pot_header(header, parent, params, state)) {
        return false;
    }

    // 6. Ed25519 miner signature must verify over the unsigned header.
    crypto::Ed25519PublicKey pubkey{};
    std::copy(header.miner_pubkey.begin(), header.miner_pubkey.end(),
              pubkey.data.begin());

    crypto::Ed25519Signature sig{};
    std::copy(header.signature.begin(), header.signature.end(),
              sig.data.begin());

    const std::vector<uint8_t> msg = header.serialize_unsigned();
    if (!crypto::ed25519_verify(pubkey,
                                std::span<const uint8_t>(msg.data(), msg.size()),
                                sig)) {
        state.invalid("bad-blk-signature");
        return false;
    }

    return true;
}

} // namespace rnet::consensus
