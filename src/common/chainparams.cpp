#include "common/chainparams.h"

#include <atomic>
#include <stdexcept>

namespace rnet::common {

static std::atomic<bool> s_selected{false};
static ChainParams s_instance;

void ChainParams::select(NetworkType type) {
    // Allow exactly one selection
    bool expected = false;
    if (!s_selected.compare_exchange_strong(expected, true)) {
        return;  // already selected — silently ignore
    }

    s_instance.type_ = type;
    switch (type) {
    case NetworkType::MAINNET:
        s_instance.consensus_ = rnet::consensus::ConsensusParams::mainnet();
        s_instance.name_ = "mainnet";
        s_instance.data_dir_ = "";
        break;
    case NetworkType::TESTNET:
        s_instance.consensus_ = rnet::consensus::ConsensusParams::testnet();
        s_instance.name_ = "testnet";
        s_instance.data_dir_ = "testnet";
        break;
    case NetworkType::REGTEST:
        s_instance.consensus_ = rnet::consensus::ConsensusParams::regtest();
        s_instance.name_ = "regtest";
        s_instance.data_dir_ = "regtest";
        break;
    }
}

const ChainParams& ChainParams::get() {
    if (!s_selected.load(std::memory_order_acquire)) {
        // Default to mainnet if not explicitly selected
        select(NetworkType::MAINNET);
    }
    return s_instance;
}

NetworkType parse_network_type(std::string_view name) {
    if (name == "testnet" || name == "test") return NetworkType::TESTNET;
    if (name == "regtest" || name == "reg")  return NetworkType::REGTEST;
    return NetworkType::MAINNET;
}

}  // namespace rnet::common
