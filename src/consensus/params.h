// Network parameters — the ResonanceNet analogue of Bitcoin's chainparams.
//
// Every consensus value lives here and nowhere else. No worker, tool or daemon
// may hardcode a model dimension, a corpus chunking or a protocol limit: they all
// read a descriptor produced from these parameters and verified by hash.
//
// A network is defined by TWO hashed artifacts, because they change for different
// reasons:
//
//   * the ROUND descriptor — what to train. Frozen until a hard fork.
//   * the POLICY descriptor — how the network schedules, verifies and settles.
//     A challenge rate or a quorum may need tuning once real data arrives, and
//     doing so must not disturb the model's identity.
//
// Three networks, same roles as Bitcoin's: main (the real training round), test
// (public experimentation, throwaway) and regtest (tiny, instant, for automated
// tests).
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "canon/objects.h"
#include "util/result.h"

namespace rnet::consensus {

enum class Network { Main, Test, Regtest };

Result<Network> ParseNetwork(std::string_view name);
std::string NetworkName(Network net);

struct NetworkParams {
    Network network{Network::Main};
    std::string name;

    // Wire magic — prefixes every P2P message so a node on the wrong network
    // disconnects immediately instead of half-parsing foreign traffic.
    std::array<uint8_t, 4> message_magic{};
    uint16_t default_port{};

    // What to train. Frozen until a hard fork.
    canon::RoundDescriptor round;

    // How the network operates. Updated independently of the round.
    canon::PolicyDescriptor policy;

    // Pinned identities of the two artifacts. These are the only values a node
    // trusts a priori — the role a hardcoded genesis block hash plays in Bitcoin.
    // Empty during bootstrap, filled once the artifacts are emitted.
    std::string genesis_hash_hex;
    std::string policy_hash_hex;

    // SHA3-256 of the canonical initial weight bytes — the state every node starts
    // from. It is an anchor for the same reason the other two are: nodes that begin
    // from different weights compute different deltas, and no amount of later
    // verification recovers from that.
    //
    // Empty until the initialization procedure is fixed and its output measured.
    // A node asked to participate without it refuses, rather than starting from
    // weights it invented — being isolated is recoverable, having trained the
    // wrong model for a week is not.
    std::string genesis_weights_hash_hex;

    // DNS seeds used to find the first peers, as in Bitcoin. Which hostnames a
    // network trusts is a property of that network, pinned here alongside its
    // genesis — not a constant in the networking code.
    std::vector<std::string> dns_seeds;

    // Literal addresses, used only when every seed is unreachable. A network
    // whose seeds can all be taken down still has to be joinable.
    std::vector<std::string> fixed_seeds;

    Status Validate() const;
};

const NetworkParams& Params(Network net);

}  // namespace rnet::consensus
