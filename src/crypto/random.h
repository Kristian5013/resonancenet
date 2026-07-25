// Randomness that an attacker must not be able to predict.
//
// Two uses in this codebase, and both are security-critical rather than
// statistical:
//
//   * THE ADDRESS-MANAGER KEY decides which bucket an address lands in. If an
//     attacker can predict it, they can craft a set of addresses that fills
//     exactly the buckets a victim selects from — which is the eclipse attack the
//     bucketing exists to prevent.
//
//   * THE HANDSHAKE NONCE is how a node recognises a connection to itself. A
//     predictable nonce lets a third party impersonate that condition and make a
//     node hang up on an honest peer.
//
// So this reads the operating system's CSPRNG and fails loudly rather than
// falling back to anything weaker. There is no seeded mode: a "deterministic
// randomness" option here would eventually be switched on in production.
#pragma once

#include <cstdint>
#include <span>

#include "crypto/sha3.h"
#include "util/result.h"

namespace rnet::crypto {

// Fills `out` from the OS CSPRNG. Fails rather than producing weak bytes.
Status RandomBytes(std::span<uint8_t> out);

// Aborts if the OS cannot provide randomness. Used where a failure has no sane
// recovery: a node that cannot get an unpredictable addrman key must not start
// with a predictable one.
Hash256 RandomHashOrDie();
uint64_t RandomU64OrDie();

}  // namespace rnet::crypto
