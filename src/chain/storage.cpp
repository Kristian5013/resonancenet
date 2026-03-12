#include "chain/storage.h"

#include <cstdio>
#include <sstream>

#include "core/logging.h"
#include "core/serialize.h"
#include "core/stream.h"

namespace rnet::chain {

std::string DiskBlockPos::to_string() const {
    std::ostringstream oss;
    oss << "DiskBlockPos(file=" << file_number << " pos=" << pos << ")";
    return oss.str();
}

BlockStorage::BlockStorage(const core::fs::path& blocks_dir)
    : blocks_dir_(blocks_dir)
{
    // Ensure the blocks directory exists
    std::error_code ec;
    core::fs::create_directories(blocks_dir_, ec);
}

BlockStorage::~BlockStorage() = default;

Result<DiskBlockPos> BlockStorage::write_block(
    const primitives::CBlock& block)
{
    LOCK(mutex_);

    // Serialize the block
    core::DataStream ss;
    block.serialize(ss);
    const auto& data = ss.vch();

    // Check if we need a new file
    if (current_pos_ + static_cast<int64_t>(data.size()) > MAX_BLOCKFILE_SIZE) {
        open_next_file();
    }

    DiskBlockPos pos;
    pos.file_number = current_file_;
    pos.pos = current_pos_;

    auto path = block_file_path(current_file_);

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

    // Write size prefix + data
    uint32_t size = static_cast<uint32_t>(data.size());
    if (std::fwrite(&size, 4, 1, f) != 1 ||
        std::fwrite(data.data(), 1, data.size(), f) != data.size()) {
        std::fclose(f);
        return Result<DiskBlockPos>::err("Failed to write block data");
    }

    std::fclose(f);

    current_pos_ += 4 + static_cast<int64_t>(data.size());

    return Result<DiskBlockPos>::ok(pos);
}

Result<primitives::CBlock> BlockStorage::read_block(
    const DiskBlockPos& pos) const
{
    if (pos.is_null()) {
        return Result<primitives::CBlock>::err("Null disk position");
    }

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

#ifdef _WIN32
    _fseeki64(f, pos.pos, SEEK_SET);
#else
    fseeko(f, pos.pos, SEEK_SET);
#endif

    // Read size prefix
    uint32_t size = 0;
    if (std::fread(&size, 4, 1, f) != 1) {
        std::fclose(f);
        return Result<primitives::CBlock>::err("Failed to read block size");
    }

    if (size > 512 * 1024 * 1024) {  // 512 MB sanity limit
        std::fclose(f);
        return Result<primitives::CBlock>::err("Block size too large");
    }

    std::vector<uint8_t> data(size);
    if (std::fread(data.data(), 1, size, f) != size) {
        std::fclose(f);
        return Result<primitives::CBlock>::err("Failed to read block data");
    }
    std::fclose(f);

    core::DataStream ss(std::move(data));
    primitives::CBlock block;
    block.unserialize(ss);

    return Result<primitives::CBlock>::ok(std::move(block));
}

Result<primitives::CBlockHeader> BlockStorage::read_block_header(
    const DiskBlockPos& pos) const
{
    // Read full block and extract header
    auto block_result = read_block(pos);
    if (!block_result) {
        return Result<primitives::CBlockHeader>::err(block_result.error());
    }
    const auto& block = block_result.value();
    return Result<primitives::CBlockHeader>::ok(
        static_cast<const primitives::CBlockHeader&>(block));
}

uint64_t BlockStorage::calculate_disk_usage() const {
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

Result<void> BlockStorage::prune_to_size(uint64_t target_bytes) {
    // Stub: in a full implementation, would remove oldest block files
    // until disk usage is below target
    (void)target_bytes;
    return Result<void>::ok();
}

core::fs::path BlockStorage::block_file_path(int file_number) const {
    char name[32];
    std::snprintf(name, sizeof(name), "blk%05d.dat", file_number);
    return blocks_dir_ / name;
}

void BlockStorage::open_next_file() {
    ++current_file_;
    current_pos_ = 0;
}

}  // namespace rnet::chain
