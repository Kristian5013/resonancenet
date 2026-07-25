// Finding the first peer.
//
// A node with no addresses has to trust something to get started, and whatever it
// trusts is the single best place to attack it: a node fed only attacker-run
// addresses is eclipsed before it has spoken to anyone honest, and everything it
// later believes about the network comes from one source.
//
// So bootstrap is treated as untrusted input, not as an oracle:
//
//   * Seeds come from CONSENSUS PARAMS, not from this file. Which hostnames a
//     network trusts is a property of that network, pinned alongside its genesis
//     — not a constant a node operator can be talked into editing.
//
//   * Every seed is queried, not the first that answers. One reachable seed
//     answering quickly must not become the node's whole view of the world.
//
//   * Results enter the address manager THROUGH THE SAME PATH as gossip from
//     peers, so they are bucketed by network group like everything else. A seed
//     returning a thousand addresses in one /16 therefore occupies one bucket,
//     not the table.
//
//   * Seed results are never marked good. Only completing a handshake proves an
//     address is worth anything, and a seed's say-so is not that proof.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "consensus/params.h"
#include "net/addrman.h"
#include "net/protocol.h"
#include "util/result.h"

namespace rnet::net {

// Addresses accepted from one seed. A seed that answers with more than this is
// either broken or trying to fill the table by itself.
inline constexpr size_t kMaxAddressesPerSeed = 256;

// Resolves one hostname to routable peer addresses. Loopback and private ranges
// are dropped: a seed pointing a node at its own machine, or into the operator's
// LAN, is not offering peers.
Result<std::vector<NetAddress>> ResolveSeed(const std::string& hostname, uint16_t default_port);

struct BootstrapResult {
    size_t seeds_queried{0};
    size_t seeds_answered{0};
    size_t addresses_added{0};
};

// Queries every seed of the network and files what they return. A seed that fails
// to resolve is not an error: networks are partly offline all the time, and a
// node that refuses to start because one hostname is down is a node that can be
// stopped by taking down one hostname.
BootstrapResult BootstrapFromSeeds(const consensus::NetworkParams& params,
                                   AddressManager& addresses, int64_t now_sec);

// Hardcoded fallbacks, used only when every seed is unreachable. Kept in the
// params rather than here for the same reason as the seeds themselves.
BootstrapResult BootstrapFromFixedSeeds(const consensus::NetworkParams& params,
                                        AddressManager& addresses, int64_t now_sec);

}  // namespace rnet::net
