#pragma once

#include <memory>
#include <string>

#include "chain/coins.h"
#include "core/error.h"
#include "core/fs.h"

namespace rnet::chain {

/// CCoinsViewDB — persistent UTXO database backed by LevelDB.
/// Keys: serialized COutPoint (txid + vout index).
/// Values: serialized Coin (CTxOut + height + is_coinbase + val_loss_at_creation).
/// Special key "B" stores the best block hash.
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

    /// Write a batch of coin updates to the database.
    /// Spent coins (is_null() on out) are deleted; live coins are written.
    /// The best block hash is updated atomically in the same batch.
    Result<void> batch_write(
        const std::unordered_map<primitives::COutPoint, Coin>& map,
        const rnet::uint256& best_block);

    /// Compact the database (triggers LevelDB compaction over full key range)
    void compact();

    /// Estimate database size on disk via LevelDB property
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
