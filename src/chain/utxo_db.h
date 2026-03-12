#pragma once

#include <memory>
#include <string>

#include "chain/coins.h"
#include "core/error.h"
#include "core/fs.h"

namespace rnet::chain {

/// CCoinsViewDB — persistent UTXO database backed by LevelDB.
/// This is a stub implementation that stores data in memory.
/// A full implementation would use LevelDB or similar.
class CCoinsViewDB : public CCoinsView {
public:
    explicit CCoinsViewDB(const core::fs::path& db_path,
                          size_t cache_bytes = 256 * 1024 * 1024);
    ~CCoinsViewDB() override;

    // Non-copyable
    CCoinsViewDB(const CCoinsViewDB&) = delete;
    CCoinsViewDB& operator=(const CCoinsViewDB&) = delete;

    bool get_coin(const primitives::COutPoint& outpoint,
                  Coin& coin) const override;

    bool have_coin(const primitives::COutPoint& outpoint) const override;

    rnet::uint256 get_best_block() const override;

    size_t estimate_size() const override;

    /// Write a batch of coin updates to the database
    Result<void> batch_write(
        const std::unordered_map<primitives::COutPoint, Coin>& map,
        const rnet::uint256& best_block);

    /// Compact the database (LevelDB compaction)
    void compact();

    /// Estimate database size on disk
    uint64_t estimated_db_size() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/// Open/create the UTXO database at the given path
Result<std::unique_ptr<CCoinsViewDB>> open_utxo_db(
    const core::fs::path& path,
    size_t cache_bytes = 256 * 1024 * 1024);

}  // namespace rnet::chain
