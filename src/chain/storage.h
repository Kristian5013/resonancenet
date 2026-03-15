#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

#include "core/error.h"
#include "core/fs.h"
#include "core/sync.h"
#include "primitives/block.h"

namespace rnet::chain {

class CBlockIndex;

/// DiskBlockPos — position of a block on disk.
struct DiskBlockPos {
    int file_number = -1;
    int64_t pos = 0;

    bool is_null() const { return file_number < 0; }
    void set_null() { file_number = -1; pos = 0; }

    std::string to_string() const;
};

/// BlockStorage — manages block data files on disk.
/// Blocks are stored in blkNNNNN.dat files (up to 128MB each).
/// Undo data is stored in revNNNNN.dat files.
class BlockStorage {
public:
    static constexpr int64_t MAX_BLOCKFILE_SIZE = 128 * 1024 * 1024;  // 128 MB

    explicit BlockStorage(const core::fs::path& blocks_dir);
    ~BlockStorage();

    // Non-copyable
    BlockStorage(const BlockStorage&) = delete;
    BlockStorage& operator=(const BlockStorage&) = delete;

    /// Write a block to disk, returning its position
    Result<DiskBlockPos> write_block(const primitives::CBlock& block);

    /// Read a block from disk given its position
    Result<primitives::CBlock> read_block(const DiskBlockPos& pos) const;

    /// Read just the block header from disk
    Result<primitives::CBlockHeader> read_block_header(
        const DiskBlockPos& pos) const;

    /// Calculate total disk usage of all block files
    uint64_t calculate_disk_usage() const;

    /// Prune old block files up to the given target size (bytes)
    Result<void> prune_to_size(uint64_t target_bytes);

    /// A block together with its disk position (for index rebuilding).
    struct StoredBlock {
        primitives::CBlock block;
        DiskBlockPos pos;
    };

    /// Scan all block files and return every stored block in order.
    /// Used at startup to rebuild the in-memory block index.
    Result<std::vector<StoredBlock>> scan_block_files() const;

    /// Get the blocks directory path
    const core::fs::path& blocks_dir() const { return blocks_dir_; }

    /// Get current file number
    int current_file() const { return current_file_; }

private:
    core::fs::path blocks_dir_;
    mutable core::Mutex mutex_;
    int current_file_ = 0;
    int64_t current_pos_ = 0;

    /// Get the path for a block file
    core::fs::path block_file_path(int file_number) const;

    /// Open the next block file when the current one is full
    void open_next_file();
};

}  // namespace rnet::chain
