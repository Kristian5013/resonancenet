// Copyright (c) 2025 The ResonanceNet developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://opensource.org/licenses/MIT.

#include "chain/storage.h"

#include "core/logging.h"
#include "core/serialize.h"
#include "core/stream.h"

#include <cstdio>
#include <sstream>

namespace rnet::chain {

// ===========================================================================
//  DiskBlockPos
// ===========================================================================

// ---------------------------------------------------------------------------
// to_string
//   Human-readable position for log output.
// ---------------------------------------------------------------------------
std::string DiskBlockPos::to_string() const
{
    std::ostringstream oss;
    oss << "DiskBlockPos(file=" << file_number << " pos=" << pos << ")";
    return oss.str();
}

// ===========================================================================
//  BlockStorage -- flat-file block persistence
// ===========================================================================

// ---------------------------------------------------------------------------
// Constructor
//   Creates the blocks directory if it does not already exist.
// ---------------------------------------------------------------------------
BlockStorage::BlockStorage(const core::fs::path& blocks_dir)
    : blocks_dir_(blocks_dir)
{
    // 1. Ensure the blocks directory exists
    std::error_code ec;
    core::fs::create_directories(blocks_dir_, ec);
}

// ---------------------------------------------------------------------------
// Destructor
// ---------------------------------------------------------------------------
BlockStorage::~BlockStorage() = default;

// ---------------------------------------------------------------------------
// write_block
//   Serialises a block, appends [4-byte size][data] to the current blk file,
//   and returns the DiskBlockPos where it was written.  Rolls to the next
//   file when MAX_BLOCKFILE_SIZE would be exceeded.
// ---------------------------------------------------------------------------
Result<DiskBlockPos> BlockStorage::write_block(
    const primitives::CBlock& block)
{
    LOCK(mutex_);

    // 1. Serialize the block
    core::DataStream ss;
    block.serialize(ss);
    const auto& data = ss.vch();

    // 2. Check if we need a new file
    if (current_pos_ + static_cast<int64_t>(data.size()) > MAX_BLOCKFILE_SIZE) {
        open_next_file();
    }

    // 3. Record position before writing
    DiskBlockPos pos;
    pos.file_number = current_file_;
    pos.pos = current_pos_;

    auto path = block_file_path(current_file_);

    // 4. Open file for appending
    FILE* f = nullptr;
#ifdef _WIN32
    fopen_s(&f, path.string().c_str(), "ab");
#else
    f = std::fopen(path.string().c_str(), "ab");
#endif
    if (!f) {
        return Result<DiskBlockPos>::err(
            "Failed to open block file: " + path.string());
    }

    // 5. Write size prefix + data
    uint32_t size = static_cast<uint32_t>(data.size());
    if (std::fwrite(&size, 4, 1, f) != 1 ||
        std::fwrite(data.data(), 1, data.size(), f) != data.size()) {
        std::fclose(f);
        return Result<DiskBlockPos>::err("Failed to write block data");
    }

    std::fclose(f);

    // 6. Advance position tracker
    current_pos_ += 4 + static_cast<int64_t>(data.size());

    return Result<DiskBlockPos>::ok(pos);
}

// ---------------------------------------------------------------------------
// read_block
//   Reads a full CBlock from the given DiskBlockPos.  Format on disk is
//   [4-byte size][serialised block].  Validates a 512 MB sanity ceiling.
// ---------------------------------------------------------------------------
Result<primitives::CBlock> BlockStorage::read_block(
    const DiskBlockPos& pos) const
{
    // 1. Reject null position
    if (pos.is_null()) {
        return Result<primitives::CBlock>::err("Null disk position");
    }

    // 2. Open the block file
    auto path = block_file_path(pos.file_number);
    FILE* f = nullptr;
#ifdef _WIN32
    fopen_s(&f, path.string().c_str(), "rb");
#else
    f = std::fopen(path.string().c_str(), "rb");
#endif
    if (!f) {
        return Result<primitives::CBlock>::err(
            "Failed to open block file: " + path.string());
    }

    // 3. Seek to the stored position
#ifdef _WIN32
    _fseeki64(f, pos.pos, SEEK_SET);
#else
    fseeko(f, pos.pos, SEEK_SET);
#endif

    // 4. Read size prefix
    uint32_t size = 0;
    if (std::fread(&size, 4, 1, f) != 1) {
        std::fclose(f);
        return Result<primitives::CBlock>::err("Failed to read block size");
    }

    // 5. Sanity-check size (512 MB ceiling)
    if (size > 512 * 1024 * 1024) {
        std::fclose(f);
        return Result<primitives::CBlock>::err("Block size too large");
    }

    // 6. Read serialised block bytes
    std::vector<uint8_t> data(size);
    if (std::fread(data.data(), 1, size, f) != size) {
        std::fclose(f);
        return Result<primitives::CBlock>::err("Failed to read block data");
    }
    std::fclose(f);

    // 7. Deserialise
    core::DataStream ss(std::move(data));
    primitives::CBlock block;
    block.unserialize(ss);

    return Result<primitives::CBlock>::ok(std::move(block));
}

// ---------------------------------------------------------------------------
// read_block_header
//   Convenience wrapper -- reads the full block and extracts just the header.
// ---------------------------------------------------------------------------
Result<primitives::CBlockHeader> BlockStorage::read_block_header(
    const DiskBlockPos& pos) const
{
    // 1. Read full block
    auto block_result = read_block(pos);
    if (!block_result) {
        return Result<primitives::CBlockHeader>::err(block_result.error());
    }

    // 2. Slice out the header
    const auto& block = block_result.value();
    return Result<primitives::CBlockHeader>::ok(
        static_cast<const primitives::CBlockHeader&>(block));
}

// ---------------------------------------------------------------------------
// calculate_disk_usage
//   Sums file sizes across all blk files [0..current_file_].
// ---------------------------------------------------------------------------
uint64_t BlockStorage::calculate_disk_usage() const
{
    uint64_t total = 0;
    for (int i = 0; i <= current_file_; ++i) {
        auto path = block_file_path(i);
        std::error_code ec;
        if (core::fs::exists(path, ec)) {
            total += core::fs::file_size(path, ec);
        }
    }
    return total;
}

// ---------------------------------------------------------------------------
// prune_to_size
//   Stub: a full implementation would remove oldest block files until disk
//   usage drops below target_bytes.
// ---------------------------------------------------------------------------
Result<void> BlockStorage::prune_to_size(uint64_t target_bytes)
{
    (void)target_bytes;
    return Result<void>::ok();
}

// ---------------------------------------------------------------------------
// block_file_path  (private)
//   Returns e.g. <blocks_dir>/blk00042.dat.
// ---------------------------------------------------------------------------
core::fs::path BlockStorage::block_file_path(int file_number) const
{
    char name[32];
    std::snprintf(name, sizeof(name), "blk%05d.dat", file_number);
    return blocks_dir_ / name;
}

// ---------------------------------------------------------------------------
// open_next_file  (private)
//   Rolls to the next numbered block file and resets the write cursor.
// ---------------------------------------------------------------------------
void BlockStorage::open_next_file()
{
    ++current_file_;
    current_pos_ = 0;
}

} // namespace rnet::chain
