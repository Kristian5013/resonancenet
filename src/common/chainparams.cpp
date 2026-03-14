// Copyright (c) 2024-present ResonanceNet developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://opensource.org/licenses/MIT.

// Own header.
#include "common/chainparams.h"

// Standard library.
#include <atomic>
#include <stdexcept>

namespace rnet::common {

static std::atomic<bool> s_selected{false};
static ChainParams s_instance;

// ---------------------------------------------------------------------------
// ChainParams::select
// ---------------------------------------------------------------------------
// Sets the global chain parameters for the given network type.  Only the
// first call takes effect; subsequent calls are silently ignored to prevent
// mid-run reconfiguration.
// ---------------------------------------------------------------------------
void ChainParams::select(NetworkType type)
{
    // 1. Allow exactly one selection.
    bool expected = false;
    if (!s_selected.compare_exchange_strong(expected, true)) {
        return;
    }

    // 2. Configure for the requested network.
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

// ---------------------------------------------------------------------------
// ChainParams::get
// ---------------------------------------------------------------------------
// Returns the global chain parameters, defaulting to mainnet if select()
// has not been called.
// ---------------------------------------------------------------------------
const ChainParams& ChainParams::get()
{
    if (!s_selected.load(std::memory_order_acquire)) {
        select(NetworkType::MAINNET);
    }
    return s_instance;
}

// ---------------------------------------------------------------------------
// parse_network_type
// ---------------------------------------------------------------------------
// Parses a string network name into the corresponding enum value.
// ---------------------------------------------------------------------------
NetworkType parse_network_type(std::string_view name)
{
    if (name == "testnet" || name == "test") return NetworkType::TESTNET;
    if (name == "regtest" || name == "reg")  return NetworkType::REGTEST;
    return NetworkType::MAINNET;
}

} // namespace rnet::common
