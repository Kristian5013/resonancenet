#pragma once

#include <cstdint>
#include <memory>
#include <unordered_map>

#include "core/error.h"
#include "core/sync.h"
#include "core/types.h"
#include "primitives/outpoint.h"
#include "primitives/txout.h"

namespace rnet::chain {

/// Coin — a UTXO entry stored in the coins database.
/// Includes the output data plus metadata for PoT UTXO expiry.
struct Coin {
    primitives::CTxOut out;      ///< The actual output (value + script)
    int height = 0;              ///< Height at which the coin was created
    bool is_coinbase = false;    ///< Whether this came from a coinbase tx
    float val_loss_at_creation = 0.0f;  ///< PoT val_loss when created (for expiry)

    Coin() = default;
    Coin(primitives::CTxOut out_in, int h, bool cb, float vl)
        : out(std::move(out_in)), height(h), is_coinbase(cb),
          val_loss_at_creation(vl) {}

    bool is_spent() const { return out.is_null(); }
};

/// CCoinsView — abstract interface for accessing the UTXO set.
class CCoinsView {
public:
    virtual ~CCoinsView() = default;

    /// Get a coin by outpoint. Returns false if not found.
    virtual bool get_coin(const primitives::COutPoint& outpoint,
                          Coin& coin) const = 0;

    /// Check if a coin exists (without fetching data)
    virtual bool have_coin(const primitives::COutPoint& outpoint) const = 0;

    /// Get the best block hash this view is synced to
    virtual rnet::uint256 get_best_block() const = 0;

    /// Get estimated UTXO set size
    virtual size_t estimate_size() const { return 0; }
};

/// CCoinsViewBacked — a view backed by another view (for layering)
class CCoinsViewBacked : public CCoinsView {
public:
    explicit CCoinsViewBacked(CCoinsView* base) : base_(base) {}

    bool get_coin(const primitives::COutPoint& outpoint,
                  Coin& coin) const override {
        return base_ ? base_->get_coin(outpoint, coin) : false;
    }

    bool have_coin(const primitives::COutPoint& outpoint) const override {
        return base_ ? base_->have_coin(outpoint) : false;
    }

    rnet::uint256 get_best_block() const override {
        return base_ ? base_->get_best_block() : rnet::uint256{};
    }

    void set_backend(CCoinsView* base) { base_ = base; }

protected:
    CCoinsView* base_ = nullptr;
};

/// CCoinsViewCache — in-memory overlay cache on top of a backing view.
/// Batch writes: modifications are collected here and flushed to the
/// backing store on flush().
class CCoinsViewCache : public CCoinsViewBacked {
public:
    explicit CCoinsViewCache(CCoinsView* base);
    ~CCoinsViewCache() override = default;

    // Non-copyable
    CCoinsViewCache(const CCoinsViewCache&) = delete;
    CCoinsViewCache& operator=(const CCoinsViewCache&) = delete;

    bool get_coin(const primitives::COutPoint& outpoint,
                  Coin& coin) const override;

    bool have_coin(const primitives::COutPoint& outpoint) const override;

    rnet::uint256 get_best_block() const override;

    size_t estimate_size() const override;

    /// Add a coin to the cache
    void add_coin(const primitives::COutPoint& outpoint, Coin coin,
                  bool possible_overwrite = false);

    /// Mark a coin as spent (removes from cache or marks spent)
    bool spend_coin(const primitives::COutPoint& outpoint,
                    Coin* moved_out = nullptr);

    /// Set the best block hash
    void set_best_block(const rnet::uint256& hash);

    /// Flush all cached changes to the backing view
    Result<void> flush();

    /// Get total cached value (for debugging)
    int64_t get_cached_value() const;

    /// Number of cached entries
    size_t cache_size() const;

    /// Check if a coin is expired based on current val_loss
    bool is_coin_expired(const Coin& coin, float current_val_loss,
                         float reclaim_ratio = 10.0f) const;

private:
    /// Fetch a coin into the local cache map (from backing store if needed)
    /// Returns iterator to the cached entry.
    using CacheMap = std::unordered_map<primitives::COutPoint, Coin>;
    mutable CacheMap cache_;
    mutable rnet::uint256 best_block_;
    mutable bool best_block_set_ = false;

    /// Fetch and cache from backing store
    CacheMap::iterator fetch_coin(
        const primitives::COutPoint& outpoint) const;
};

}  // namespace rnet::chain
