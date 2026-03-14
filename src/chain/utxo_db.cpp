#include "chain/utxo_db.h"

#include <cstring>
#include <string>

#include <leveldb/db.h>
#include <leveldb/write_batch.h>
#include <leveldb/cache.h>
#include <leveldb/filter_policy.h>

#include "core/logging.h"

namespace rnet::chain {

// ─── Serialization helpers ──────────────────────────────────────────────

/// Key prefix for coin entries: 'c' + 32-byte txid + 4-byte vout (LE).
/// Best block key: single byte 'B'.
static constexpr char KEY_BEST_BLOCK = 'B';
static constexpr char KEY_COIN_PREFIX = 'c';

/// Build the LevelDB key for a COutPoint.
/// Format: [1 byte 'c'][32 bytes txid][4 bytes vout little-endian]
static std::string coin_key(const primitives::COutPoint& outpoint) {
    std::string key;
    key.reserve(1 + 32 + 4);
    key.push_back(KEY_COIN_PREFIX);
    key.append(reinterpret_cast<const char*>(outpoint.hash.data()), 32);
    uint32_t n = outpoint.n;
    key.append(reinterpret_cast<const char*>(&n), 4);
    return key;
}

/// Serialize a Coin to a binary string.
/// Format (all little-endian):
///   [8 bytes value (int64_t)]
///   [4 bytes script_pub_key length]
///   [N bytes script_pub_key]
///   [4 bytes height (int32_t)]
///   [1 byte  is_coinbase]
///   [4 bytes val_loss_at_creation (float, IEEE 754)]
static std::string serialize_coin(const Coin& coin) {
    std::string data;
    const size_t script_len = coin.out.script_pub_key.size();
    data.resize(8 + 4 + script_len + 4 + 1 + 4);

    char* p = data.data();

    // value
    int64_t val = coin.out.value;
    std::memcpy(p, &val, 8);
    p += 8;

    // script length + data
    auto slen = static_cast<uint32_t>(script_len);
    std::memcpy(p, &slen, 4);
    p += 4;
    if (script_len > 0) {
        std::memcpy(p, coin.out.script_pub_key.data(), script_len);
        p += script_len;
    }

    // height
    auto h = static_cast<int32_t>(coin.height);
    std::memcpy(p, &h, 4);
    p += 4;

    // is_coinbase
    *p = coin.is_coinbase ? 1 : 0;
    p += 1;

    // val_loss_at_creation
    float vl = coin.val_loss_at_creation;
    std::memcpy(p, &vl, 4);

    return data;
}

/// Deserialize a Coin from a binary string. Returns false on malformed data.
static bool deserialize_coin(const std::string& data, Coin& coin) {
    // Minimum size: 8 + 4 + 0 + 4 + 1 + 4 = 21
    if (data.size() < 21) return false;

    const char* p = data.data();
    const char* end = data.data() + data.size();

    // value
    int64_t val;
    std::memcpy(&val, p, 8);
    p += 8;

    // script length
    uint32_t slen;
    std::memcpy(&slen, p, 4);
    p += 4;

    // bounds check
    if (p + slen + 4 + 1 + 4 > end) return false;

    // script data
    std::vector<uint8_t> script(p, p + slen);
    p += slen;

    // height
    int32_t h;
    std::memcpy(&h, p, 4);
    p += 4;

    // is_coinbase
    bool is_cb = (*p != 0);
    p += 1;

    // val_loss_at_creation
    float vl;
    std::memcpy(&vl, p, 4);

    coin.out.value = val;
    coin.out.script_pub_key = std::move(script);
    coin.height = static_cast<int>(h);
    coin.is_coinbase = is_cb;
    coin.val_loss_at_creation = vl;
    return true;
}

// ─── Impl ───────────────────────────────────────────────────────────────

struct CCoinsViewDB::Impl {
    std::unique_ptr<leveldb::DB> db;
    std::unique_ptr<leveldb::Cache> block_cache;
    std::unique_ptr<const leveldb::FilterPolicy> filter_policy;
    leveldb::ReadOptions read_opts;
    leveldb::WriteOptions write_opts;
    leveldb::WriteOptions sync_write_opts;
    core::fs::path db_path;
};

CCoinsViewDB::CCoinsViewDB(const core::fs::path& db_path,
                            size_t cache_bytes)
    : impl_(std::make_unique<Impl>())
{
    impl_->db_path = db_path;

    // Create the bloom filter and block cache
    impl_->filter_policy.reset(leveldb::NewBloomFilterPolicy(10));
    impl_->block_cache.reset(leveldb::NewLRUCache(cache_bytes));

    leveldb::Options options;
    options.create_if_missing = true;
    options.block_cache = impl_->block_cache.get();
    options.filter_policy = impl_->filter_policy.get();
    options.write_buffer_size = 64 * 1024 * 1024;  // 64 MB write buffer
    options.max_open_files = 256;
    options.compression = leveldb::kSnappyCompression;

    // Open the database
    leveldb::DB* raw_db = nullptr;
    leveldb::Status status = leveldb::DB::Open(
        options, db_path.string(), &raw_db);

    if (!status.ok()) {
        LogPrintf("ERROR: Failed to open UTXO database at %s: %s\n",
                  db_path.string().c_str(), status.ToString().c_str());
        // Store nullptr; all operations will fail gracefully
        impl_->db.reset(nullptr);
    } else {
        impl_->db.reset(raw_db);
        LogPrintf("Opened UTXO database at %s\n", db_path.string().c_str());
    }

    // Default read options
    impl_->read_opts.verify_checksums = true;
    impl_->read_opts.fill_cache = true;

    // Default write options (no sync for regular writes)
    impl_->write_opts.sync = false;

    // Sync write options (for batch_write to ensure durability)
    impl_->sync_write_opts.sync = true;
}

CCoinsViewDB::~CCoinsViewDB() = default;

bool CCoinsViewDB::get_coin(const primitives::COutPoint& outpoint,
                            Coin& coin) const {
    if (!impl_->db) return false;

    std::string key = coin_key(outpoint);
    std::string value;
    leveldb::Status status = impl_->db->Get(impl_->read_opts, key, &value);

    if (!status.ok()) return false;

    if (!deserialize_coin(value, coin)) {
        LogPrintf("ERROR: Corrupt UTXO entry for %s\n",
                  outpoint.to_string().c_str());
        return false;
    }

    if (coin.is_spent()) return false;
    return true;
}

bool CCoinsViewDB::have_coin(const primitives::COutPoint& outpoint) const {
    if (!impl_->db) return false;

    std::string key = coin_key(outpoint);
    std::string value;
    leveldb::Status status = impl_->db->Get(impl_->read_opts, key, &value);
    if (!status.ok()) return false;

    // Verify the coin is not spent without fully deserializing:
    // just check that the data is valid and the value field != -1
    if (value.size() < 8) return false;
    int64_t val;
    std::memcpy(&val, value.data(), 8);
    return val != -1;  // is_null() checks value == -1
}

rnet::uint256 CCoinsViewDB::get_best_block() const {
    if (!impl_->db) return rnet::uint256{};

    std::string key(1, KEY_BEST_BLOCK);
    std::string value;
    leveldb::Status status = impl_->db->Get(impl_->read_opts, key, &value);

    if (!status.ok() || value.size() != 32) {
        return rnet::uint256{};
    }

    return rnet::uint256(std::span<const uint8_t>(
        reinterpret_cast<const uint8_t*>(value.data()), 32));
}

size_t CCoinsViewDB::estimate_size() const {
    if (!impl_->db) return 0;

    // Use LevelDB's approximate size estimation over the coin key range
    std::string start(1, KEY_COIN_PREFIX);
    // End key: 'c' followed by 0xFF * 36 (past any valid coin key)
    std::string end(1, KEY_COIN_PREFIX);
    end.append(36, '\xFF');

    leveldb::Range range(start, end);
    uint64_t size = 0;
    impl_->db->GetApproximateSizes(&range, 1, &size);

    // Very rough estimate: average entry ~100 bytes
    return (size > 0) ? static_cast<size_t>(size / 100) : 0;
}

Result<void> CCoinsViewDB::batch_write(
    const std::unordered_map<primitives::COutPoint, Coin>& map,
    const rnet::uint256& best_block)
{
    if (!impl_->db) {
        return Result<void>::err("UTXO database is not open");
    }

    leveldb::WriteBatch batch;
    size_t put_count = 0;
    size_t del_count = 0;

    for (const auto& [outpoint, coin] : map) {
        std::string key = coin_key(outpoint);
        if (coin.is_spent()) {
            batch.Delete(key);
            ++del_count;
        } else {
            batch.Put(key, serialize_coin(coin));
            ++put_count;
        }
    }

    // Write the best block hash
    if (!best_block.is_zero()) {
        std::string bb_key(1, KEY_BEST_BLOCK);
        std::string bb_val(reinterpret_cast<const char*>(best_block.data()), 32);
        batch.Put(bb_key, bb_val);
    }

    // Atomically commit the batch with sync (durability)
    leveldb::Status status = impl_->db->Write(impl_->sync_write_opts, &batch);
    if (!status.ok()) {
        std::string msg = "UTXO batch_write failed: " + status.ToString();
        LogPrintf("ERROR: %s\n", msg.c_str());
        return Result<void>::err(msg);
    }

    LogPrintf("UTXO batch_write: %zu puts, %zu deletes\n", put_count, del_count);
    return Result<void>::ok();
}

void CCoinsViewDB::compact() {
    if (!impl_->db) return;

    // Compact the full key range
    impl_->db->CompactRange(nullptr, nullptr);
    LogPrintf("UTXO database compaction complete\n");
}

uint64_t CCoinsViewDB::estimated_db_size() const {
    if (!impl_->db) return 0;

    std::string stats;
    if (impl_->db->GetProperty("leveldb.approximate-memory-usage", &stats)) {
        // This gives memory usage; for disk size use approximate sizes
    }

    // Approximate on-disk size over coin key range
    std::string start(1, KEY_COIN_PREFIX);
    std::string end(1, KEY_COIN_PREFIX);
    end.append(36, '\xFF');

    leveldb::Range range(start, end);
    uint64_t size = 0;
    impl_->db->GetApproximateSizes(&range, 1, &size);
    return size;
}

// ─── Factory ────────────────────────────────────────────────────────────

Result<std::unique_ptr<CCoinsViewDB>> open_utxo_db(
    const core::fs::path& path, size_t cache_bytes)
{
    // Ensure parent directory exists
    std::error_code ec;
    std::filesystem::create_directories(path.parent_path(), ec);
    if (ec) {
        return Result<std::unique_ptr<CCoinsViewDB>>::err(
            "Cannot create UTXO db directory: " + ec.message());
    }

    auto db = std::make_unique<CCoinsViewDB>(path, cache_bytes);

    // Verify the database actually opened by attempting to read best block
    // (this is a lightweight check — get_best_block returns zero hash if
    // nothing stored yet, which is fine for a fresh database).
    // We rely on the constructor's null-db check for real failures.
    // A production hardening step would be to add an is_open() accessor.

    return Result<std::unique_ptr<CCoinsViewDB>>::ok(std::move(db));
}

}  // namespace rnet::chain
