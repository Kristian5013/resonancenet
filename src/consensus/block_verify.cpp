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

// ----------------------------------------------------------------------------
// Script sigop counting — walks opcodes, skipping pushed data
// ----------------------------------------------------------------------------

static int count_script_sigops(const std::vector<uint8_t>& script) {
    int sigops = 0;
    size_t i = 0;
    while (i < script.size()) {
        uint8_t opcode = script[i];

        // 1. Direct push (1-75 bytes): skip opcode + N data bytes
        if (opcode >= 1 && opcode <= 75) {
            i += 1 + opcode;
            continue;
        }

        // 2. OP_PUSHDATA1 (0x4C): next byte is length
        if (opcode == 0x4C) {
            if (i + 1 >= script.size()) break;
            uint8_t len = script[i + 1];
            i += 2 + len;
            continue;
        }

        // 3. OP_PUSHDATA2 (0x4D): next two bytes are length (little-endian)
        if (opcode == 0x4D) {
            if (i + 2 >= script.size()) break;
            uint16_t len = static_cast<uint16_t>(script[i + 1])
                         | (static_cast<uint16_t>(script[i + 2]) << 8);
            i += 3 + len;
            continue;
        }

        // 4. OP_CHECKSIG (0xAC) / OP_CHECKSIGVERIFY (0xAD)
        if (opcode == 0xAC || opcode == 0xAD) {
            sigops += 1;
        }

        // 5. OP_CHECKMULTISIG (0xAE) / OP_CHECKMULTISIGVERIFY (0xAF)
        if (opcode == 0xAE || opcode == 0xAF) {
            sigops += 20;
        }

        ++i;
    }
    return sigops;
}

// ----------------------------------------------------------------------------
// Block validation
// ----------------------------------------------------------------------------

bool check_block(const primitives::CBlock& block,
                 ValidationState& state,
                 const ConsensusParams& params) {
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

    // --- Block size limit ------------------------------------------------
    size_t block_size = block.get_block_size();
    if (block_size > params.max_block_size) {
        state.invalid("bad-blk-size");
        return false;
    }

    // --- Sigop count limit -----------------------------------------------
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

    // --- Validate each transaction ---------------------------------------
    for (const auto& tx : block.vtx) {
        if (!check_transaction(*tx, state)) {
            return false;
        }
    }

    return true;
}

// ----------------------------------------------------------------------------
// Block header validation
// ----------------------------------------------------------------------------

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

    // --- Structural checks -----------------------------------------------

    // 1. Height must be parent + 1
    if (header.height != parent.height + 1) {
        state.invalid("bad-height");
        return false;
    }

    // 2. prev_hash must match parent's hash
    if (!(header.prev_hash == parent.hash())) {
        state.invalid("bad-prevblk");
        return false;
    }

    // 3. Timestamp must be strictly greater than parent
    if (header.timestamp <= parent.timestamp) {
        state.invalid("bad-timestamp");
        return false;
    }

    // 4. Reject blocks more than 2 hours in the future
    if (header.timestamp > core::get_time() + 7200) {
        state.invalid("bad-timestamp-future");
        return false;
    }

    // --- Proof-of-Training -----------------------------------------------

    // 5. Verify PoT header fields
    if (!verify_pot_header(header, parent, params, state)) {
        return false;
    }

    // --- Block signature -------------------------------------------------

    // 6. Verify Ed25519 signature over unsigned header data
    crypto::Ed25519PublicKey pubkey{};
    std::copy(header.miner_pubkey.begin(), header.miner_pubkey.end(),
              pubkey.data.begin());

    crypto::Ed25519Signature sig{};
    std::copy(header.signature.begin(), header.signature.end(),
              sig.data.begin());

    std::vector<uint8_t> msg = header.serialize_unsigned();
    if (!crypto::ed25519_verify(pubkey,
                                std::span<const uint8_t>(msg.data(), msg.size()),
                                sig)) {
        state.invalid("bad-blk-signature");
        return false;
    }

    return true;
}

}  // namespace rnet::consensus
