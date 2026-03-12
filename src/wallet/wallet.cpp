#include "wallet/wallet.h"

#include "core/logging.h"
#include "wallet/sign_tx.h"
#include "wallet/spend.h"

#include <chrono>

namespace rnet::wallet {

Result<std::unique_ptr<CWallet>> CWallet::create(
    const std::filesystem::path& path,
    const std::string& name,
    primitives::NetworkType network,
    script::RecoveryType recovery_type,
    const script::RecoveryPolicy& recovery_policy) {

    auto wallet = std::unique_ptr<CWallet>(new CWallet());

    // Set mandatory recovery policy FIRST
    auto rec_result = wallet->recovery_.set_policy(recovery_type, recovery_policy);
    if (!rec_result) {
        return Result<std::unique_ptr<CWallet>>::err(
            "wallet creation requires valid recovery policy: " + rec_result.error());
    }

    wallet->network_ = network;
    wallet->metadata_.name = name;
    wallet->metadata_.creation_time = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    wallet->metadata_.version = 1;

    // Map network type
    switch (network) {
        case primitives::NetworkType::MAINNET: wallet->metadata_.network = "mainnet"; break;
        case primitives::NetworkType::TESTNET: wallet->metadata_.network = "testnet"; break;
        case primitives::NetworkType::REGTEST: wallet->metadata_.network = "regtest"; break;
    }

    // Create HD wallet
    auto mnemonic_result = wallet->hd_.create();
    if (!mnemonic_result) {
        return Result<std::unique_ptr<CWallet>>::err(
            "HD wallet creation failed: " + mnemonic_result.error());
    }

    // Generate initial receive address
    auto addr_result = wallet->get_new_address("default");
    if (!addr_result) {
        return Result<std::unique_ptr<CWallet>>::err(
            "initial address generation failed: " + addr_result.error());
    }

    // Open database
    wallet->db_ = std::make_unique<WalletDB>(path);
    auto db_result = wallet->db_->open();
    if (!db_result) {
        return Result<std::unique_ptr<CWallet>>::err(
            "database open failed: " + db_result.error());
    }

    // Save initial state
    auto save_result = wallet->save();
    if (!save_result) {
        return Result<std::unique_ptr<CWallet>>::err(
            "initial save failed: " + save_result.error());
    }

    LogPrint(WALLET, "wallet created: %s", name.c_str());
    return Result<std::unique_ptr<CWallet>>::ok(std::move(wallet));
}

Result<std::unique_ptr<CWallet>> CWallet::load(
    const std::filesystem::path& path) {

    auto wallet = std::unique_ptr<CWallet>(new CWallet());

    wallet->db_ = std::make_unique<WalletDB>(path);
    auto db_result = wallet->db_->open();
    if (!db_result) {
        return Result<std::unique_ptr<CWallet>>::err(
            "database open failed: " + db_result.error());
    }

    // Read data
    std::vector<WalletKey> keys;
    HDState hd_state;
    std::vector<AddressEntry> addresses;
    std::vector<WalletCoin> coins;
    std::vector<uint8_t> encrypt_data;
    std::vector<uint8_t> recovery_data;

    auto read_result = wallet->db_->read(
        wallet->metadata_, keys, hd_state, addresses, coins,
        encrypt_data, recovery_data);
    if (!read_result) {
        return Result<std::unique_ptr<CWallet>>::err(
            "database read failed: " + read_result.error());
    }

    // Restore keys
    for (const auto& key : keys) {
        wallet->keys_.add_key(key);
    }

    // Restore HD state
    if (!hd_state.mnemonic.empty()) {
        wallet->hd_.set_state(std::move(hd_state));
    }

    // Restore addresses
    for (const auto& addr : addresses) {
        wallet->addresses_.add_entry(addr);
    }

    // Restore coins
    for (const auto& coin : coins) {
        wallet->coins_.add_coin(coin);
    }

    // Restore encryption state
    if (!encrypt_data.empty()) {
        wallet->encryptor_.set_encrypted_key(encrypt_data);
    }

    // Determine network
    if (wallet->metadata_.network == "testnet") {
        wallet->network_ = primitives::NetworkType::TESTNET;
    } else if (wallet->metadata_.network == "regtest") {
        wallet->network_ = primitives::NetworkType::REGTEST;
    } else {
        wallet->network_ = primitives::NetworkType::MAINNET;
    }

    LogPrint(WALLET, "wallet loaded: %s (%zu keys, %zu addresses, %zu coins)",
             wallet->metadata_.name.c_str(),
             wallet->keys_.size(),
             wallet->addresses_.size(),
             wallet->coins_.unspent_count());

    return Result<std::unique_ptr<CWallet>>::ok(std::move(wallet));
}

CWallet::~CWallet() {
    if (db_) {
        save();
        db_->close();
    }
}

// ─── Key management ─────────────────────────────────────────────────

Result<std::string> CWallet::get_mnemonic() const {
    return hd_.get_mnemonic();
}

Result<std::string> CWallet::get_new_address(const std::string& label) {
    LOCK(mutex_);

    // Derive next external key
    auto key_result = hd_.derive_next_external();
    if (!key_result) {
        return Result<std::string>::err("key derivation failed: " + key_result.error());
    }
    auto& key = key_result.value();

    // Add key to store
    auto add_result = keys_.add_key(key);
    if (!add_result) {
        return Result<std::string>::err("key store add failed: " + add_result.error());
    }

    // Create address
    return addresses_.create_address(key.pubkey_hash, label, false, network_);
}

Result<std::string> CWallet::get_new_change_address() {
    LOCK(mutex_);

    auto key_result = hd_.derive_next_internal();
    if (!key_result) {
        return Result<std::string>::err("key derivation failed: " + key_result.error());
    }
    auto& key = key_result.value();

    auto add_result = keys_.add_key(key);
    if (!add_result) {
        return Result<std::string>::err("key store add failed: " + add_result.error());
    }

    return addresses_.create_address(key.pubkey_hash, "", true, network_);
}

bool CWallet::is_mine(const std::string& address) const {
    return addresses_.is_mine(address);
}

bool CWallet::is_mine_hash(const uint160& pubkey_hash) const {
    return addresses_.is_mine_hash(pubkey_hash);
}

// ─── Balance ────────────────────────────────────────────────────────

WalletBalance CWallet::get_balance(int32_t current_height) const {
    return compute_balance(coins_, current_height);
}

// ─── UTXO management ────────────────────────────────────────────────

Result<void> CWallet::add_coin(const WalletCoin& coin) {
    auto result = coins_.add_coin(coin);
    if (result) {
        notifier_.notify_tx_received(coin.outpoint.hash, coin.txout.value, "");
    }
    return result;
}

Result<void> CWallet::spend_coin(const primitives::COutPoint& outpoint) {
    return coins_.spend_coin(outpoint);
}

std::vector<WalletCoin> CWallet::get_unspent_coins() const {
    return coins_.get_unspent();
}

// ─── Transaction creation ───────────────────────────────────────────

Result<primitives::CTransaction> CWallet::send_to(
    const std::string& address,
    int64_t amount,
    primitives::FeeEstimateTarget fee_target) {

    std::vector<Recipient> recipients;
    recipients.push_back({address, amount, false});

    auto create_result = create_transaction(recipients, fee_target);
    if (!create_result) {
        return Result<primitives::CTransaction>::err(create_result.error());
    }
    auto& ctx_result = create_result.value();

    // Sign
    int signed_count = sign_wallet_transaction(
        keys_, ctx_result.tx, ctx_result.coin_selection.selected);
    if (signed_count == 0) {
        return Result<primitives::CTransaction>::err("signing failed");
    }

    // Mark spent coins
    for (const auto& coin : ctx_result.coin_selection.selected) {
        coins_.spend_coin(coin.outpoint);
    }

    auto tx = primitives::CTransaction(std::move(ctx_result.tx));
    notifier_.notify_tx_sent(tx.txid(), amount);

    return Result<primitives::CTransaction>::ok(std::move(tx));
}

Result<CreateTxResult> CWallet::create_transaction(
    const std::vector<Recipient>& recipients,
    primitives::FeeEstimateTarget fee_target) {

    LOCK(mutex_);

    auto unspent = coins_.get_unspent();
    if (unspent.empty()) {
        return Result<CreateTxResult>::err("no unspent coins");
    }

    // Get or create change address
    auto change_result = get_new_change_address();
    if (!change_result) {
        return Result<CreateTxResult>::err(
            "change address failed: " + change_result.error());
    }

    // Get fee rate
    auto fee_est = fee_estimator_.estimate(fee_target);

    CreateTxParams params;
    params.recipients = recipients;
    params.fee_rate = fee_est.rate;
    params.change_address = change_result.value();
    params.network = network_;
    params.rbf = true;

    // Find the change pubkey hash
    auto change_entry = addresses_.get_by_address(change_result.value());
    if (change_entry) {
        params.change_pubkey_hash = change_entry.value().pubkey_hash;
    }

    return wallet::create_transaction(unspent, params);
}

// ─── Heartbeat ──────────────────────────────────────────────────────

Result<primitives::CTransaction> CWallet::create_heartbeat() {
    LOCK(mutex_);

    auto unspent = coins_.get_unspent();
    if (unspent.empty()) {
        return Result<primitives::CTransaction>::err("no coins for heartbeat");
    }

    // Use the first address's pubkey hash
    auto all_hashes = keys_.get_all_pubkey_hashes();
    if (all_hashes.empty()) {
        return Result<primitives::CTransaction>::err("no keys available");
    }

    auto hb_result = HeartbeatCreator::create_heartbeat_tx(
        unspent, all_hashes[0]);
    if (!hb_result) {
        return Result<primitives::CTransaction>::err(hb_result.error());
    }

    // Find the spent coin for signing
    std::vector<WalletCoin> spent_coins;
    for (const auto& txin : hb_result.value().vin) {
        auto coin_result = coins_.get_coin(txin.prevout);
        if (coin_result) {
            spent_coins.push_back(coin_result.value());
        }
    }

    auto sign_result = HeartbeatCreator::sign_heartbeat_tx(
        keys_, hb_result.value(), spent_coins);
    if (!sign_result) {
        return Result<primitives::CTransaction>::err(sign_result.error());
    }

    // Mark spent
    for (const auto& coin : spent_coins) {
        coins_.spend_coin(coin.outpoint);
    }

    return Result<primitives::CTransaction>::ok(
        primitives::CTransaction(std::move(hb_result.value())));
}

bool CWallet::heartbeat_due(uint64_t blocks_since_last) const {
    return recovery_.heartbeat_due(blocks_since_last);
}

// ─── Encryption ─────────────────────────────────────────────────────

Result<void> CWallet::encrypt_wallet(const std::string& passphrase) {
    return encryptor_.encrypt(passphrase);
}

Result<void> CWallet::unlock(const std::string& passphrase) {
    return encryptor_.unlock(passphrase);
}

void CWallet::lock() {
    encryptor_.lock();
}

bool CWallet::is_encrypted() const {
    return encryptor_.is_encrypted();
}

bool CWallet::is_locked() const {
    return encryptor_.is_encrypted() && !encryptor_.is_unlocked();
}

Result<void> CWallet::change_passphrase(const std::string& old_pass,
                                        const std::string& new_pass) {
    return encryptor_.change_passphrase(old_pass, new_pass);
}

// ─── Persistence ────────────────────────────────────────────────────

Result<void> CWallet::save() {
    if (!db_ || !db_->is_open()) {
        return Result<void>::err("database not open");
    }

    auto all_addresses = addresses_.get_all();
    auto all_coins = coins_.get_all();

    return db_->write(
        metadata_,
        keys_,
        hd_.state(),
        all_addresses,
        all_coins,
        encryptor_.get_encrypted_key(),
        std::vector<uint8_t>{}  // Recovery data (TODO: serialize policy)
    );
}

const std::filesystem::path& CWallet::path() const {
    return db_->path();
}

// ─── Notifications ──────────────────────────────────────────────────

size_t CWallet::register_notify(WalletNotifyCallback cb) {
    return notifier_.register_callback(std::move(cb));
}

void CWallet::unregister_notify(size_t handle) {
    notifier_.unregister_callback(handle);
}

}  // namespace rnet::wallet
