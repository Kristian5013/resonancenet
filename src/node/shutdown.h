#pragma once

namespace rnet::node {

/// Request a global shutdown (thread-safe, can be called from signal handler)
void request_shutdown();

/// Check whether shutdown has been requested
bool shutdown_requested();

/// Block the calling thread until shutdown is requested.
/// Uses a condition variable — does not spin.
void wait_for_shutdown();

/// Reset the shutdown flag (for test harnesses that restart the node)
void reset_shutdown();

} // namespace rnet::node
