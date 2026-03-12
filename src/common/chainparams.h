#pragma once

#include "consensus/params.h"

#include <string>
#include <string_view>

namespace rnet::common {

enum class NetworkType {
    MAINNET,
    TESTNET,
    REGTEST,
};

/// Global chain parameters — selected at startup, immutable after
class ChainParams {
public:
    static void select(NetworkType type);
    static const ChainParams& get();

    const rnet::consensus::ConsensusParams& consensus() const { return consensus_; }
    NetworkType network_type() const { return type_; }

    const std::string& network_name() const { return name_; }
    const std::string& bech32_hrp() const { return consensus_.bech32_hrp; }
    uint16_t default_port() const { return consensus_.default_port; }
    uint16_t rpc_port() const { return consensus_.rpc_port; }
    uint16_t lightning_port() const { return consensus_.lightning_port; }

    const std::string& data_dir_name() const { return data_dir_; }

    ChainParams() = default;

private:
    rnet::consensus::ConsensusParams consensus_;
    NetworkType type_ = NetworkType::MAINNET;
    std::string name_;
    std::string data_dir_;
};

/// Parse network name from string ("mainnet", "testnet", "regtest")
NetworkType parse_network_type(std::string_view name);

}  // namespace rnet::common
