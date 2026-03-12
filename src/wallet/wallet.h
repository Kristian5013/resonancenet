#pragma once

#include <filesystem>
#include <memory>
#include <string>

#include "core/error.h"
#include "core/sync.h"
#include "core/types.h"
#include "primitives/address.h"
#include "primitives/transaction.h"
#include "wallet/addresses.h"
#include "wallet/balance.h"
#include "wallet/coins.h"
#include "wallet/create_tx.h"
#include "wallet/encrypt.h"
#include "wallet/fees.h"
#include "wallet/hd.h"
#include "wallet/heartbeat.h"
#include "wallet/keys.h"
#include "wallet/notify.h"
#include "wallet/recovery.h"
#include "wallet/walletdb.h"

namespace rnet::wallet {

/// CWallet: main wallet class.
///
/// Manages keys, addresses, UTXOs, balance, transaction creation/signing,
/// encryption, HD derivation, recovery policies, and heartbeat transactions.
///
/// IMPORTANT: wallet creation FAILS without a mandatory recovery policy.
class CWallet {
public:
    /// Create a new wallet.
    /// @param path       Database file path.
    /// @param name       Wallet name.
    /// @param network    Network type for address encoding.
    /// @param policy     MANDATORY recovery policy (creation fails without one).
    static Result<std::unique_ptr<CWallet>> create(
        const std::filesystem::path& path,
        const std::string& name,
        primitives::NetworkType network,
        script::RecoveryType recovery_type,
        const script::RecoveryPolicy& recovery_policy);

    /// Load an existing wallet from disk.
    static Result<std::unique_ptr<CWallet>> load(
        const std::filesystem::path& path);

    ~CWallet();

    CWallet(const CWallet&) = delete;
    CWallet& operator=(const CWallet&) = delete;

    // ─── Key management ─────────────────────────────────────────────

    /// Get the HD mnemonic (for backup display). Requires unlocked wallet.
    Result<std::string> get_mnemonic() const;

    /// Generate a new receive address.
    Result<std::string> get_new_address(const std::string& label = "");

    /// Generate a new change address (internal chain).
    Result<std::string> get_new_change_address();

    /// Check if an address belongs to this wallet.
    bool is_mine(const std::string& address) const;

    /// Check if a pubkey hash belongs to this wallet.
    bool is_mine_hash(const uint160& pubkey_hash) const;

    // ─── Balance ────────────────────────────────────────────────────

    /// Get wallet balance.
    WalletBalance get_balance(int32_t current_height) const;

    // ─── UTXO management ────────────────────────────────────────────

    /// Add a received coin (called when scanning blocks/mempool).
    Result<void> add_coin(const WalletCoin& coin);

    /// Mark a coin as spent.
    Result<void> spend_coin(const primitives::COutPoint& outpoint);

    /// Get all unspent coins.
    std::vector<WalletCoin> get_unspent_coins() const;

    // ─── Transaction creation ───────────────────────────────────────

    /// Create and sign a transaction.
    Result<primitives::CTransaction> send_to(
        const std::string& address,
        int64_t amount,
        primitives::FeeEstimateTarget fee_target =
            primitives::FeeEstimateTarget::CONSERVATIVE);

    /// Create an unsigned transaction (for advanced use).
    Result<CreateTxResult> create_transaction(
        const std::vector<Recipient>& recipients,
        primitives::FeeEstimateTarget fee_target =
            primitives::FeeEstimateTarget::CONSERVATIVE);

    // ─── Heartbeat ──────────────────────────────────────────────────

    /// Create and sign a heartbeat transaction.
    Result<primitives::CTransaction> create_heartbeat();

    /// Check if heartbeat is due.
    bool heartbeat_due(uint64_t blocks_since_last) const;

    // ─── Encryption ─────────────────────────────────────────────────

    /// Encrypt the wallet with a passphrase.
    Result<void> encrypt_wallet(const std::string& passphrase);

    /// Unlock an encrypted wallet.
    Result<void> unlock(const std::string& passphrase);

    /// Lock the wallet.
    void lock();

    /// Check if wallet is encrypted.
    bool is_encrypted() const;

    /// Check if wallet is locked.
    bool is_locked() const;

    /// Change the encryption passphrase.
    Result<void> change_passphrase(const std::string& old_pass,
                                   const std::string& new_pass);

    // ─── Persistence ────────────────────────────────────────────────

    /// Save wallet state to disk.
    Result<void> save();

    /// Get the wallet name.
    const std::string& name() const { return metadata_.name; }

    /// Get the wallet path.
    const std::filesystem::path& path() const;

    // ─── Notifications ──────────────────────────────────────────────

    /// Register a notification callback.
    size_t register_notify(WalletNotifyCallback cb);

    /// Unregister a notification callback.
    void unregister_notify(size_t handle);

    // ─── Recovery ───────────────────────────────────────────────────

    /// Get the recovery manager.
    const RecoveryManager& recovery() const { return recovery_; }

    // ─── Fee estimation ─────────────────────────────────────────────

    /// Get the fee estimator.
    FeeEstimator& fee_estimator() { return fee_estimator_; }
    const FeeEstimator& fee_estimator() const { return fee_estimator_; }

    // ─── Components (for advanced use) ──────────────────────────────

    KeyStore& key_store() { return keys_; }
    const KeyStore& key_store() const { return keys_; }
    HDKeyManager& hd_manager() { return hd_; }
    AddressManager& address_manager() { return addresses_; }
    CoinTracker& coin_tracker() { return coins_; }
    WalletNotifier& notifier() { return notifier_; }

private:
    CWallet() = default;

    mutable core::Mutex mutex_;

    // Components
    KeyStore keys_;
    HDKeyManager hd_;
    AddressManager addresses_;
    CoinTracker coins_;
    WalletEncryptor encryptor_;
    FeeEstimator fee_estimator_;
    RecoveryManager recovery_;
    WalletNotifier notifier_;
    WalletMetadata metadata_;
    std::unique_ptr<WalletDB> db_;
    primitives::NetworkType network_ = primitives::NetworkType::MAINNET;
};

}  // namespace rnet::wallet
