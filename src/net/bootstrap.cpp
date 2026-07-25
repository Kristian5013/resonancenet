#include "net/bootstrap.h"

#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>

#include <cstring>

#include "crypto/sha3.h"
#include "util/logging.h"

namespace rnet::net {

namespace {

// The source recorded for a seed's addresses. Bucketing is keyed by source as
// well as by address, so this is what stops one seed's answer from spreading
// across the whole table.
NetAddress SeedSource(const std::string& hostname, uint16_t port) {
    NetAddress source;
    // A synthetic, stable address per hostname: distinct seeds land in distinct
    // buckets, and the same seed lands in the same bucket every time.
    const auto digest = crypto::Sha3_256(hostname);
    static constexpr uint8_t kV4Mapped[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xFF, 0xFF};
    std::memcpy(source.ip.data(), kV4Mapped, 12);
    std::memcpy(source.ip.data() + 12, digest.data(), 4);
    source.port = port;
    return source;
}

}  // namespace

Result<std::vector<NetAddress>> ResolveSeed(const std::string& hostname, uint16_t default_port) {
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;         // both families; a network reachable only
    hints.ai_socktype = SOCK_STREAM;     // over one of them is still reachable
    hints.ai_protocol = IPPROTO_TCP;

    addrinfo* results = nullptr;
    const int rc = ::getaddrinfo(hostname.c_str(), nullptr, &hints, &results);
    if (rc != 0) {
        return Err(std::string("seed: ") + hostname + ": " + ::gai_strerror(rc));
    }

    std::vector<NetAddress> addresses;
    for (addrinfo* entry = results; entry != nullptr; entry = entry->ai_next) {
        if (addresses.size() >= kMaxAddressesPerSeed) break;

        NetAddress address;
        if (entry->ai_family == AF_INET) {
            static constexpr uint8_t kV4Mapped[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xFF, 0xFF};
            std::memcpy(address.ip.data(), kV4Mapped, 12);
            const auto* in4 = reinterpret_cast<const sockaddr_in*>(entry->ai_addr);
            std::memcpy(address.ip.data() + 12, &in4->sin_addr, 4);
        } else if (entry->ai_family == AF_INET6) {
            const auto* in6 = reinterpret_cast<const sockaddr_in6*>(entry->ai_addr);
            std::memcpy(address.ip.data(), &in6->sin6_addr, 16);
        } else {
            continue;
        }
        address.port = default_port;

        // A seed pointing a node at loopback or into a private range is not
        // offering peers, whether by misconfiguration or on purpose.
        if (!address.IsRoutable()) continue;
        addresses.push_back(address);
    }
    ::freeaddrinfo(results);

    if (addresses.empty()) return Err("seed: " + hostname + ": no routable addresses");
    return addresses;
}

BootstrapResult BootstrapFromSeeds(const consensus::NetworkParams& params,
                                   AddressManager& addresses, int64_t now_sec) {
    BootstrapResult result;

    // Every seed is queried. Stopping at the first that answers would let one
    // reachable seed become the node's entire view of the network.
    for (const auto& hostname : params.dns_seeds) {
        ++result.seeds_queried;

        // Said before the call, not after. getaddrinfo blocks, and a name that
        // does not resolve costs the resolver's full timeout — twenty seconds is
        // typical. Without this line the node looks hung during a wait that is
        // both normal and explainable, which is how an operator concludes the
        // software is broken and stops it halfway.
        RNET_LOG_INFO("bootstrap: resolving {}", hostname);
        auto resolved = ResolveSeed(hostname, params.default_port);
        if (!resolved) {
            // Networks are partly offline all the time. A node that refuses to
            // start because one hostname is down can be stopped by taking down
            // one hostname.
            RNET_LOG_DEBUG("{}", resolved.error());
            continue;
        }
        ++result.seeds_answered;

        // Filed through the same path as gossip from peers, so a seed's answer is
        // bucketed by network group like everything else — and never marked good,
        // because only a completed handshake proves an address is worth anything.
        const size_t added =
            addresses.AddMany(resolved.value(), SeedSource(hostname, params.default_port), now_sec);
        result.addresses_added += added;
        RNET_LOG_INFO("seed {} offered {} addresses, {} new", hostname, resolved.value().size(),
                      added);
    }
    return result;
}

BootstrapResult BootstrapFromFixedSeeds(const consensus::NetworkParams& params,
                                        AddressManager& addresses, int64_t now_sec) {
    BootstrapResult result;
    if (params.fixed_seeds.empty()) return result;

    std::vector<NetAddress> parsed;
    for (const auto& text : params.fixed_seeds) {
        auto address = NetAddress::FromString(text, params.default_port);
        if (!address) {
            RNET_LOG_DEBUG("fixed seed '{}' is unparseable: {}", text, address.error());
            continue;
        }
        if (!address.value().IsRoutable()) continue;
        parsed.push_back(address.value());
    }
    if (parsed.empty()) return result;

    result.seeds_queried = params.fixed_seeds.size();
    result.seeds_answered = parsed.size();
    result.addresses_added = addresses.AddMany(parsed, SeedSource("fixed", params.default_port),
                                               now_sec);
    return result;
}

}  // namespace rnet::net
