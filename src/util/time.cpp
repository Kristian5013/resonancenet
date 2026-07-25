#include "util/time.h"

#include <chrono>

namespace rnet::util {

int64_t NowMillis() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

int64_t NowSeconds() { return NowMillis() / 1000; }

int64_t MonotonicMillis() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

}  // namespace rnet::util
