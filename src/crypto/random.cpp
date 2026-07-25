#include "crypto/random.h"

#include <sys/random.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace rnet::crypto {

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
