// Wall-clock time, in one place.
//
// The node loop takes `now_ms` as a parameter rather than reading a clock, so
// that tests are deterministic and two nodes cannot disagree because of clock
// skew. This header is where that parameter comes from at the top of the process
// — nowhere else in the library reads a clock.
#pragma once

#include <cstdint>

namespace rnet::util {

// Milliseconds since the Unix epoch. Used for protocol timestamps, which must be
// comparable between machines, so this is wall time rather than a monotonic
// counter.
int64_t NowMillis();

// Seconds since the Unix epoch — the unit the address database ages entries in.
int64_t NowSeconds();

// Monotonic milliseconds, for measuring durations. Immune to the clock being
// stepped, which wall time is not: an operator running ntpdate must not be able
// to make every peer look like it timed out.
int64_t MonotonicMillis();

}  // namespace rnet::util
