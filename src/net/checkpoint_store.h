#pragma once

#include <cstdint>
#include <filesystem>
#include <vector>

#include "core/error.h"
#include "core/sync.h"
#include "core/types.h"

namespace rnet::net {

/// CheckpointStore — file-based storage for model checkpoints.
///
/// Stores checkpoints as flat files under <datadir>/checkpoints/<hash_hex>.rnet.
/// Thread-safe: all public methods acquire cs_store_ internally.
class CheckpointStore {
public:
    /// Construct a store rooted at the given directory.
    /// Creates the directory if it does not exist.
    explicit CheckpointStore(const std::filesystem::path& data_dir);
    ~CheckpointStore();

    // Non-copyable
    CheckpointStore(const CheckpointStore&) = delete;
    CheckpointStore& operator=(const CheckpointStore&) = delete;

    // ── Queries ──────────────────────────────────────────────────────

    /// Check if a checkpoint exists locally.
    bool has(const rnet::uint256& hash) const;

    /// Load a checkpoint from disk.
    Result<std::vector<uint8_t>> load(const rnet::uint256& hash) const;

    /// Save a checkpoint to disk.
    Result<void> save(const rnet::uint256& hash,
                      const std::vector<uint8_t>& data);

    /// Get the filesystem path for a given checkpoint hash.
    std::filesystem::path path_for(const rnet::uint256& hash) const;

    /// Get the root directory.
    const std::filesystem::path& root() const { return checkpoint_dir_; }

private:
    std::filesystem::path checkpoint_dir_;
    mutable core::Mutex cs_store_;
};

}  // namespace rnet::net
