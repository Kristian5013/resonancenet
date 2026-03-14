#include "consensus/params.h"

namespace rnet::consensus {

ConsensusParams ConsensusParams::mainnet() {
    ConsensusParams p;
    // All defaults are mainnet values — nothing to override.
    // consensus_dataset_hash: will be set when mainnet dataset is pinned.
    return p;
}

ConsensusParams ConsensusParams::testnet() {
    ConsensusParams p;
    p.magic = {0x54, 0x4E, 0x45, 0x54};  // "TNET"
    p.default_port = 19555;
    p.rpc_port = 19554;
    p.lightning_port = 19556;
    p.bech32_hrp = "tn";

    // Lower thresholds for faster testing
    p.halving_thresholds = {
        static_cast<int64_t>(525'000) * primitives::COIN,
        static_cast<int64_t>(1'050'000) * primitives::COIN,
        static_cast<int64_t>(1'575'000) * primitives::COIN,
        static_cast<int64_t>(1'837'500) * primitives::COIN,
    };

    // Smaller model limits
    p.max_d_model = 2048;
    p.max_layers = 24;

    // consensus_dataset_hash: will be set when testnet dataset is pinned.

    return p;
}

ConsensusParams ConsensusParams::regtest() {
    ConsensusParams p;
    p.magic = {0x52, 0x45, 0x47, 0x54};  // "REGT"
    p.default_port = 29555;
    p.rpc_port = 29554;
    p.lightning_port = 29556;
    p.bech32_hrp = "rr";

    // Fast parameters for integration testing
    p.initial_reward = 50 * primitives::COIN;
    p.halving_thresholds = {
        static_cast<int64_t>(1'000) * primitives::COIN,
        static_cast<int64_t>(2'000) * primitives::COIN,
        static_cast<int64_t>(3'000) * primitives::COIN,
        static_cast<int64_t>(3'500) * primitives::COIN,
    };

    // Minimal PoT requirements
    p.min_steps_per_block = 1;
    p.max_steps_per_block = 1000;
    p.eval_batches = 2;
    p.loss_verify_tolerance = 0.10f;

    // Small growth
    p.base_growth = 2;
    p.growth_patience = 3;
    p.layer_threshold = 32;
    p.max_d_model = 512;
    p.max_layers = 12;

    // Quick heartbeat
    p.min_heartbeat_interval = 100;
    p.max_heartbeat_interval = 5000;
    p.recovery_waiting_period = 10;
    p.emission_inactivity_period = 2000;

    // Smaller blocks
    p.max_block_size = 10'000'000;
    p.max_checkpoint_size = 50'000'000;
    p.max_block_sigops = 20000;
    p.finality_depth = 2;

    // No dataset pinning in regtest — consensus_dataset_hash stays zero.

    return p;
}

}  // namespace rnet::consensus
