#include "consensus/genesis.h"
#include "consensus/merkle.h"

#include "primitives/transaction.h"
#include "primitives/txin.h"
#include "primitives/txout.h"
#include "primitives/outpoint.h"

#include <cstring>
#include <string>

namespace rnet::consensus {

primitives::CBlock create_genesis_block(const ConsensusParams& params) {
    primitives::CBlock block;

    // --- Header ---
    block.version = 1;
    block.height = 0;
    block.prev_hash = rnet::uint256{};  // zero
    block.timestamp = 1741824000;       // 2025-03-13 00:00:00 UTC (epoch anchor)

    // PoT fields for genesis
    block.val_loss = 10.0f;
    block.prev_val_loss = 0.0f;
    block.train_steps = 0;

    // Model config at genesis
    block.d_model = params.genesis_d_model;
    block.n_layers = params.genesis_n_layers;
    block.n_slots = params.genesis_n_slots;
    block.d_ff = 2 * params.genesis_d_model;
    block.vocab_size = params.genesis_vocab_size;
    block.max_seq_len = 2048;

    // Growth tracking: no growth at genesis
    block.stagnation_count = 0;
    block.growth_delta = 0;

    // Miner pubkey and signature are zero (genesis is unsigned)
    block.miner_pubkey = {};
    block.signature = {};

    // --- Coinbase transaction ---
    primitives::CMutableTransaction mtx;
    mtx.version = 1;

    // Coinbase input: null outpoint with genesis message as scriptSig
    primitives::COutPoint null_outpoint;
    null_outpoint.set_null();

    std::string msg(GENESIS_MESSAGE);
    std::vector<uint8_t> script_sig(msg.begin(), msg.end());

    primitives::CTxIn coinbase_in(null_outpoint, std::move(script_sig), 0xFFFFFFFF);
    mtx.vin.push_back(std::move(coinbase_in));

    // Coinbase output: initial reward to a dummy script
    // (genesis coins are typically unspendable)
    // Using Ed25519 coinbase script format: [0x20][32 zero bytes][0xAC]
    std::vector<uint8_t> genesis_script;
    genesis_script.push_back(0x20);  // push 32 bytes
    genesis_script.resize(33, 0x00); // 32 zero bytes
    genesis_script.push_back(0xAC);  // OP_CHECKSIG

    primitives::CTxOut coinbase_out(params.initial_reward, std::move(genesis_script));
    mtx.vout.push_back(std::move(coinbase_out));

    mtx.locktime = 0;

    block.vtx.push_back(primitives::MakeTransactionRef(std::move(mtx)));

    // Compute and set merkle root
    block.merkle_root = block_merkle_root(block);

    return block;
}

}  // namespace rnet::consensus
