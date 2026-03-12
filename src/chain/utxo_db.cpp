#include "chain/utxo_db.h"

#include <unordered_map>

namespace rnet::chain {

/// Stub implementation using an in-memory map.
/// Replace with LevelDB when available.
struct CCoinsViewDB::Impl {
    core::fs::path db_path;
    size_t cache_bytes = 0;
    std::unordered_map<primitives::COutPoint, Coin> store;
    rnet::uint256 best_block;
};

CCoinsViewDB::CCoinsViewDB(const core::fs::path& db_path,
                            size_t cache_bytes)
    : impl_(std::make_unique<Impl>())
{
    impl_->db_path = db_path;
    impl_->cache_bytes = cache_bytes;
}

CCoinsViewDB::~CCoinsViewDB() = default;

bool CCoinsViewDB::get_coin(const primitives::COutPoint& outpoint,
                            Coin& coin) const {
    auto it = impl_->store.find(outpoint);
    if (it == impl_->store.end()) return false;
    if (it->second.is_spent()) return false;
    coin = it->second;
    return true;
}

bool CCoinsViewDB::have_coin(const primitives::COutPoint& outpoint) const {
    auto it = impl_->store.find(outpoint);
    return it != impl_->store.end() && !it->second.is_spent();
}

rnet::uint256 CCoinsViewDB::get_best_block() const {
    return impl_->best_block;
}

size_t CCoinsViewDB::estimate_size() const {
    return impl_->store.size();
}

Result<void> CCoinsViewDB::batch_write(
    const std::unordered_map<primitives::COutPoint, Coin>& map,
    const rnet::uint256& best_block)
{
    for (const auto& [op, coin] : map) {
        if (coin.is_spent()) {
            impl_->store.erase(op);
        } else {
            impl_->store[op] = coin;
        }
    }
    impl_->best_block = best_block;
    return Result<void>::ok();
}

void CCoinsViewDB::compact() {
    // No-op for in-memory stub
}

uint64_t CCoinsViewDB::estimated_db_size() const {
    // Rough estimate: each entry ~200 bytes
    return impl_->store.size() * 200;
}

Result<std::unique_ptr<CCoinsViewDB>> open_utxo_db(
    const core::fs::path& path, size_t cache_bytes)
{
    auto db = std::make_unique<CCoinsViewDB>(path, cache_bytes);
    return Result<std::unique_ptr<CCoinsViewDB>>::ok(std::move(db));
}

}  // namespace rnet::chain
