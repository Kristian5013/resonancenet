// Copyright (c) 2024-2026 The ResonanceNet Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "chain/utxo_db.h"

#include "core/logging.h"

#include <cstring>
#include <string>

#include <leveldb/cache.h>
#include <leveldb/db.h>
#include <leveldb/filter_policy.h>
#include <leveldb/write_batch.h>

namespace rnet::chain {

// Key format constants
//
// Coin key:       'c' || txid (32 bytes) || vout (4 bytes LE)
// Best block key: 'B' (single byte)
static constexpr char KEY_BEST_BLOCK  = 'B';
static constexpr char KEY_COIN_PREFIX = 'c';

// ---------------------------------------------------------------------------
// coin_key
// ---------------------------------------------------------------------------
// Build the LevelDB key for a COutPoint.
// Layout: [1 byte 'c'][32 bytes txid][4 bytes vout little-endian]  (37 total)
// ---------------------------------------------------------------------------
static std::string coin_key(const primitives::COutPoint& outpoint) {
    std::string key;
    key.reserve(1 + 32 + 4);

    // 1. Prefix byte.
    key.push_back(KEY_COIN_PREFIX);

    // 2. Raw txid hash bytes.
    key.append(reinterpret_cast<const char*>(outpoint.hash.data()), 32);

    // 3. Output index (little-endian uint32).
    uint32_t n = outpoint.n;
    key.append(reinterpret_cast<const char*>(&n), 4);

    return key;
}

// ---------------------------------------------------------------------------
// serialize_coin
// ---------------------------------------------------------------------------
// Binary coin format (all little-endian):
//
//   Offset   Size   Field
//   ------   ----   -----
//   0        8      value (int64_t)
//   8        4      script_pub_key length (uint32_t)
//   12       N      script_pub_key bytes
//   12+N     4      height (int32_t)
//   16+N     1      is_coinbase (0 or 1)
//   17+N     4      val_loss_at_creation (float, IEEE 754)
//
// Minimum size: 21 bytes (empty script).
// ---------------------------------------------------------------------------
static std::string serialize_coin(const Coin& coin) {
    const size_t script_len = coin.out.script_pub_key.size();
    std::string data;
    data.resize(8 + 4 + script_len + 4 + 1 + 4);

    char* p = data.data();

    // 1. Value.
    int64_t val = coin.out.value;
    std::memcpy(p, &val, 8);
    p += 8;

    // 2. Script length + data.
    auto slen = static_cast<uint32_t>(script_len);
    std::memcpy(p, &slen, 4);
    p += 4;
    if (script_len > 0) {
        std::memcpy(p, coin.out.script_pub_key.data(), script_len);
        p += script_len;
    }

    // 3. Height.
    auto h = static_cast<int32_t>(coin.height);
    std::memcpy(p, &h, 4);
    p += 4;

    // 4. Coinbase flag.
    *p = coin.is_coinbase ? 1 : 0;
    p += 1;

    // 5. Validation-loss snapshot.
    float vl = coin.val_loss_at_creation;
    std::memcpy(p, &vl, 4);

    return data;
}

// ---------------------------------------------------------------------------
// deserialize_coin
// ---------------------------------------------------------------------------
// Inverse of serialize_coin.  Returns false on malformed data.
// ---------------------------------------------------------------------------
static bool deserialize_coin(const std::string& data, Coin& coin) {
    // 1. Minimum-size check (empty script ⇒ 8+4+0+4+1+4 = 21).
    if (data.size() < 21) return false;

    const char* p   = data.data();
    const char* end = data.data() + data.size();

    // 2. Value.
    int64_t val;
    std::memcpy(&val, p, 8);
    p += 8;

    // 3. Script length.
    uint32_t slen;
    std::memcpy(&slen, p, 4);
    p += 4;

    // 4. Bounds check — remaining bytes must cover script + tail fields.
    if (p + slen + 4 + 1 + 4 > end) return false;

    // 5. Script data.
    std::vector<uint8_t> script(p, p + slen);
    p += slen;

    // 6. Height.
    int32_t h;
    std::memcpy(&h, p, 4);
    p += 4;

    // 7. Coinbase flag.
    bool is_cb = (*p != 0);
    p += 1;

    // 8. Validation-loss snapshot.
    float vl;
    std::memcpy(&vl, p, 4);

    coin.out.value            = val;
    coin.out.script_pub_key   = std::move(script);
    coin.height               = static_cast<int>(h);
    coin.is_coinbase          = is_cb;
    coin.val_loss_at_creation = vl;
    return true;
}

// ---------------------------------------------------------------------------
// CCoinsViewDB::Impl
// ---------------------------------------------------------------------------
struct CCoinsViewDB::Impl {
    std::unique_ptr<leveldb::DB>                 db;
    std::unique_ptr<leveldb::Cache>              block_cache;
    std::unique_ptr<const leveldb::FilterPolicy> filter_policy;
    leveldb::ReadOptions                         read_opts;
    leveldb::WriteOptions                        write_opts;
    leveldb::WriteOptions                        sync_write_opts;
    core::fs::path                               db_path;
};

// ---------------------------------------------------------------------------
// CCoinsViewDB::CCoinsViewDB
// ---------------------------------------------------------------------------
CCoinsViewDB::CCoinsViewDB(const core::fs::path& db_path,
                            size_t cache_bytes)
    : impl_(std::make_unique<Impl>())
{
    impl_->db_path = db_path;

    // 1. Bloom filter + block cache.
    impl_->filter_policy.reset(leveldb::NewBloomFilterPolicy(10));
    impl_->block_cache.reset(leveldb::NewLRUCache(cache_bytes));

    // 2. LevelDB options.
    leveldb::Options options;
    options.create_if_missing = true;
    options.block_cache       = impl_->block_cache.get();
    options.filter_policy     = impl_->filter_policy.get();
    options.write_buffer_size = 64 * 1024 * 1024;  // 64 MB write buffer
    options.max_open_files    = 256;
    options.compression       = leveldb::kSnappyCompression;

    // 3. Open the database.
    leveldb::DB* raw_db = nullptr;
    leveldb::Status status = leveldb::DB::Open(
        options, db_path.string(), &raw_db);

    if (!status.ok()) {
        LogPrintf("ERROR: Failed to open UTXO database at %s: %s\n",
                  db_path.string().c_str(), status.ToString().c_str());
        impl_->db.reset(nullptr);
    } else {
        impl_->db.reset(raw_db);
        LogPrintf("Opened UTXO database at %s\n", db_path.string().c_str());
    }

    // 4. Read options — verify checksums, populate cache.
    impl_->read_opts.verify_checksums = true;
    impl_->read_opts.fill_cache       = true;

    // 5. Write options — async for regular writes, sync for batch commits.
    impl_->write_opts.sync      = false;
    impl_->sync_write_opts.sync = true;
}

// ---------------------------------------------------------------------------
// CCoinsViewDB::~CCoinsViewDB
// ---------------------------------------------------------------------------
CCoinsViewDB::~CCoinsViewDB() = default;

// ---------------------------------------------------------------------------
// CCoinsViewDB::get_coin
// ---------------------------------------------------------------------------
bool CCoinsViewDB::get_coin(const primitives::COutPoint& outpoint,
                            Coin& coin) const {
    if (!impl_->db) return false;

    // 1. Look up the raw LevelDB entry.
    std::string key = coin_key(outpoint);
    std::string value;
    leveldb::Status status = impl_->db->Get(impl_->read_opts, key, &value);
    if (!status.ok()) return false;

    // 2. Deserialize.
    if (!deserialize_coin(value, coin)) {
        LogPrintf("ERROR: Corrupt UTXO entry for %s\n",
                  outpoint.to_string().c_str());
        return false;
    }

    // 3. Reject spent coins.
    if (coin.is_spent()) return false;

    return true;
}

// ---------------------------------------------------------------------------
// CCoinsViewDB::have_coin
// ---------------------------------------------------------------------------
bool CCoinsViewDB::have_coin(const primitives::COutPoint& outpoint) const {
    if (!impl_->db) return false;

    // 1. Fetch raw bytes.
    std::string key = coin_key(outpoint);
    std::string value;
    leveldb::Status status = impl_->db->Get(impl_->read_opts, key, &value);
    if (!status.ok()) return false;

    // 2. Quick spent check — value == -1 means is_null() / spent.
    if (value.size() < 8) return false;
    int64_t val;
    std::memcpy(&val, value.data(), 8);
    return val != -1;
}

// ---------------------------------------------------------------------------
// CCoinsViewDB::get_best_block
// ---------------------------------------------------------------------------
rnet::uint256 CCoinsViewDB::get_best_block() const {
    if (!impl_->db) return rnet::uint256{};

    // 1. Read the single-byte 'B' key.
    std::string key(1, KEY_BEST_BLOCK);
    std::string value;
    leveldb::Status status = impl_->db->Get(impl_->read_opts, key, &value);

    if (!status.ok() || value.size() != 32) {
        return rnet::uint256{};
    }

    // 2. Reinterpret the raw 32 bytes as uint256.
    return rnet::uint256(std::span<const uint8_t>(
        reinterpret_cast<const uint8_t*>(value.data()), 32));
}

// ---------------------------------------------------------------------------
// CCoinsViewDB::estimate_size
// ---------------------------------------------------------------------------
size_t CCoinsViewDB::estimate_size() const {
    if (!impl_->db) return 0;

    // 1. Define the coin key range ['c' ... 'c' + 0xFF*36].
    std::string start(1, KEY_COIN_PREFIX);
    std::string end(1, KEY_COIN_PREFIX);
    end.append(36, '\xFF');

    // 2. Ask LevelDB for an approximate byte count.
    leveldb::Range range(start, end);
    uint64_t size = 0;
    impl_->db->GetApproximateSizes(&range, 1, &size);

    // 3. Rough coin count — assume ~100 bytes per entry.
    return (size > 0) ? static_cast<size_t>(size / 100) : 0;
}

// ---------------------------------------------------------------------------
// CCoinsViewDB::batch_write
// ---------------------------------------------------------------------------
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

    // 1. Populate the batch — delete spent coins, put live ones.
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

    // 2. Update the best-block hash (skip if zero — no change).
    if (!best_block.is_zero()) {
        std::string bb_key(1, KEY_BEST_BLOCK);
        std::string bb_val(reinterpret_cast<const char*>(best_block.data()), 32);
        batch.Put(bb_key, bb_val);
    }

    // 3. Atomic, synchronous commit.
    leveldb::Status status = impl_->db->Write(impl_->sync_write_opts, &batch);
    if (!status.ok()) {
        std::string msg = "UTXO batch_write failed: " + status.ToString();
        LogPrintf("ERROR: %s\n", msg.c_str());
        return Result<void>::err(msg);
    }

    LogPrintf("UTXO batch_write: %zu puts, %zu deletes\n", put_count, del_count);
    return Result<void>::ok();
}

// ---------------------------------------------------------------------------
// CCoinsViewDB::compact
// ---------------------------------------------------------------------------
void CCoinsViewDB::compact() {
    if (!impl_->db) return;

    // 1. Compact the full key range.
    impl_->db->CompactRange(nullptr, nullptr);
    LogPrintf("UTXO database compaction complete\n");
}

// ---------------------------------------------------------------------------
// CCoinsViewDB::estimated_db_size
// ---------------------------------------------------------------------------
uint64_t CCoinsViewDB::estimated_db_size() const {
    if (!impl_->db) return 0;

    // 1. Query memory stats (currently unused, kept for diagnostics).
    std::string stats;
    impl_->db->GetProperty("leveldb.approximate-memory-usage", &stats);

    // 2. Approximate on-disk size over coin key range.
    std::string start(1, KEY_COIN_PREFIX);
    std::string end(1, KEY_COIN_PREFIX);
    end.append(36, '\xFF');

    leveldb::Range range(start, end);
    uint64_t size = 0;
    impl_->db->GetApproximateSizes(&range, 1, &size);
    return size;
}

// ---------------------------------------------------------------------------
// open_utxo_db
// ---------------------------------------------------------------------------
Result<std::unique_ptr<CCoinsViewDB>> open_utxo_db(
    const core::fs::path& path, size_t cache_bytes)
{
    // 1. Ensure parent directory exists.
    std::error_code ec;
    std::filesystem::create_directories(path.parent_path(), ec);
    if (ec) {
        return Result<std::unique_ptr<CCoinsViewDB>>::err(
            "Cannot create UTXO db directory: " + ec.message());
    }

    // 2. Construct the database object (opens LevelDB internally).
    auto db = std::make_unique<CCoinsViewDB>(path, cache_bytes);

    return Result<std::unique_ptr<CCoinsViewDB>>::ok(std::move(db));
}

} // namespace rnet::chain
