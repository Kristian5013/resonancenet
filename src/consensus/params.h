#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "core/types.h"
#include "primitives/amount.h"

namespace rnet::consensus {

/// ConsensusParams — all consensus-critical constants in one place.
/// Network-specific presets via static factory methods.
struct ConsensusParams {
    // --- Network identification ---
    std::array<uint8_t, 4> magic = {0x52, 0x4E, 0x45, 0x54};  // "RNET"
    uint16_t default_port = 9555;
    uint16_t rpc_port = 9554;
    uint16_t lightning_port = 9556;
    std::string bech32_hrp = "rn";

    // --- Economics ---
    int64_t initial_reward = 50 * primitives::COIN;
    int64_t max_supply = static_cast<int64_t>(21'000'000) * primitives::COIN;
    std::vector<int64_t> halving_thresholds = {
        static_cast<int64_t>(5'250'000) * primitives::COIN,
        static_cast<int64_t>(10'500'000) * primitives::COIN,
        static_cast<int64_t>(15'750'000) * primitives::COIN,
        static_cast<int64_t>(18'375'000) * primitives::COIN,
    };

    // --- Proof-of-Training ---
    float loss_verify_tolerance = 0.02f;
    int min_steps_per_block = 100;
    int max_steps_per_block = 50000;
    int eval_batches = 20;

    /// Consensus-pinned dataset hash. Blocks must train on a dataset matching
    /// this hash. Zero means "no dataset pinning" (used in regtest).
    rnet::uint256 consensus_dataset_hash{};

    // --- Genesis model configuration ---
    uint32_t genesis_d_model = 384;
    uint32_t genesis_n_layers = 6;
    uint32_t genesis_n_slots = 64;
    uint32_t genesis_vocab_size = 50257;

    // --- Continuous growth constants ---
    uint32_t base_growth = 2;
    uint32_t growth_patience = 10;
    uint32_t layer_threshold = 128;
    uint32_t max_d_model = 4096;
    uint32_t max_layers = 48;

    // --- Recovery / heartbeat policies ---
    uint64_t min_heartbeat_interval = 10000;
    uint64_t max_heartbeat_interval = 500000;
    uint64_t recovery_waiting_period = 1000;
    uint64_t emission_inactivity_period = 200000;

    // --- Block rules ---
    uint64_t max_block_size = 300'000'000;
    uint64_t max_checkpoint_size = 500'000'000;
    int max_block_sigops = 80000;
    int finality_depth = 6;

    // --- Factory methods for network presets ---
    static ConsensusParams mainnet();
    static ConsensusParams testnet();
    static ConsensusParams regtest();
};

}  // namespace rnet::consensus
