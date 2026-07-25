#include "crypto/random.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <format>

#ifdef _WIN32
#include <windows.h>
// After windows.h, which bcrypt.h requires.
#include <bcrypt.h>
#else
#include <sys/random.h>
#endif

namespace rnet::crypto {

// Two implementations of one requirement: bytes an attacker cannot predict, or a
// loud failure. Neither falls back to anything weaker — a seeded generator here
// would eventually be switched on in production, and the addrman key it produces
// decides which bucket every address lands in.

#ifdef _WIN32

Status RandomBytes(std::span<uint8_t> out) {
    if (out.empty()) return Status::Ok();
    // BCRYPT_USE_SYSTEM_PREFERRED_RNG asks for the OS generator rather than an
    // algorithm handle this process opened, so there is no configuration through
    // which a caller could weaken it.
    const NTSTATUS status = ::BCryptGenRandom(nullptr, reinterpret_cast<PUCHAR>(out.data()),
                                              static_cast<ULONG>(out.size()),
                                              BCRYPT_USE_SYSTEM_PREFERRED_RNG);
    if (status < 0) {
        return Err(std::format("BCryptGenRandom failed with 0x{:08x}",
                               static_cast<unsigned long>(status)));
    }
    return Status::Ok();
}

#else

Status RandomBytes(std::span<uint8_t> out) {
    size_t filled = 0;
    while (filled < out.size()) {
        // getrandom can return fewer bytes than asked for, and can be interrupted.
        // Treating a short read as success is how a buffer ends up half-zero.
        const ssize_t got = ::getrandom(out.data() + filled, out.size() - filled, 0);
        if (got < 0) {
            if (errno == EINTR) continue;
            return Err(std::string("getrandom: ") + std::strerror(errno));
        }
        if (got == 0) return Err("getrandom returned no bytes");
        filled += static_cast<size_t>(got);
    }
    return Status::Ok();
}

#endif

Hash256 RandomHashOrDie() {
    Hash256 out{};
    if (auto st = RandomBytes(out); !st) {
        std::fprintf(stderr, "fatal: no system randomness available: %s\n", st.error().c_str());
        std::abort();
    }
    return out;
}

uint64_t RandomU64OrDie() {
    uint8_t bytes[8];
    if (auto st = RandomBytes(bytes); !st) {
        std::fprintf(stderr, "fatal: no system randomness available: %s\n", st.error().c_str());
        std::abort();
    }
    uint64_t value = 0;
    for (uint8_t b : bytes) value = (value << 8) | b;
    return value;
}

}  // namespace rnet::crypto
