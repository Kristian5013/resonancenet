#include "net/checkpoint_store.h"

#include <fstream>

#include "core/logging.h"

namespace rnet::net {

// ---------------------------------------------------------------------------
// Construction / Destruction
// ---------------------------------------------------------------------------

CheckpointStore::CheckpointStore(const std::filesystem::path& data_dir)
    : checkpoint_dir_(data_dir / "checkpoints")
{
    // 1. Ensure the checkpoints directory exists
    std::error_code ec;
    std::filesystem::create_directories(checkpoint_dir_, ec);
    if (ec) {
        LogError("CheckpointStore: failed to create directory %s: %s",
                 checkpoint_dir_.string().c_str(), ec.message().c_str());
    }
}

CheckpointStore::~CheckpointStore() = default;

// ---------------------------------------------------------------------------
// Queries
// ---------------------------------------------------------------------------

bool CheckpointStore::has(const rnet::uint256& hash) const {
    LOCK(cs_store_);
    return std::filesystem::exists(path_for(hash));
}

Result<std::vector<uint8_t>> CheckpointStore::load(
    const rnet::uint256& hash) const {
    LOCK(cs_store_);

    // 1. Build the file path
    auto fpath = path_for(hash);

    // 2. Open the file
    std::ifstream file(fpath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        return Result<std::vector<uint8_t>>::err(
            "Checkpoint not found: " + hash.to_hex().substr(0, 16));
    }

    // 3. Read size and allocate buffer
    auto size = file.tellg();
    if (size <= 0) {
        return Result<std::vector<uint8_t>>::err(
            "Checkpoint file empty: " + hash.to_hex().substr(0, 16));
    }

    std::vector<uint8_t> data(static_cast<size_t>(size));
    file.seekg(0, std::ios::beg);
    file.read(reinterpret_cast<char*>(data.data()),
              static_cast<std::streamsize>(data.size()));

    if (!file.good()) {
        return Result<std::vector<uint8_t>>::err(
            "Checkpoint read error: " + hash.to_hex().substr(0, 16));
    }

    LogDebug(NET, "Loaded checkpoint %s (%zu bytes)",
             hash.to_hex().substr(0, 16).c_str(), data.size());

    return Result<std::vector<uint8_t>>::ok(std::move(data));
}

Result<void> CheckpointStore::save(const rnet::uint256& hash,
                                   const std::vector<uint8_t>& data) {
    LOCK(cs_store_);

    // 1. Build the file path
    auto fpath = path_for(hash);

    // 2. Skip if already stored
    if (std::filesystem::exists(fpath)) {
        LogDebug(NET, "Checkpoint %s already stored, skipping",
                 hash.to_hex().substr(0, 16).c_str());
        return Result<void>::ok();
    }

    // 3. Write to a temp file, then rename for atomicity
    auto tmp_path = fpath;
    tmp_path += ".tmp";

    std::ofstream file(tmp_path, std::ios::binary | std::ios::trunc);
    if (!file.is_open()) {
        return Result<void>::err(
            "Cannot open checkpoint file for writing: " +
            tmp_path.string());
    }

    file.write(reinterpret_cast<const char*>(data.data()),
               static_cast<std::streamsize>(data.size()));
    file.flush();

    if (!file.good()) {
        std::filesystem::remove(tmp_path);
        return Result<void>::err(
            "Checkpoint write error: " + hash.to_hex().substr(0, 16));
    }

    file.close();

    // 4. Rename temp file to final path
    std::error_code ec;
    std::filesystem::rename(tmp_path, fpath, ec);
    if (ec) {
        std::filesystem::remove(tmp_path);
        return Result<void>::err(
            "Checkpoint rename failed: " + ec.message());
    }

    LogPrint(NET, "Saved checkpoint %s (%zu bytes)",
             hash.to_hex().substr(0, 16).c_str(), data.size());

    return Result<void>::ok();
}

std::filesystem::path CheckpointStore::path_for(
    const rnet::uint256& hash) const {
    return checkpoint_dir_ / (hash.to_hex() + ".rnet");
}

}  // namespace rnet::net
