#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "core/error.h"
#include "core/sync.h"
#include "core/types.h"
#include "crypto/bip32.h"
#include "crypto/bip39.h"
#include "wallet/keys.h"

namespace rnet::wallet {

/// BIP44 derivation path: m/44'/9555'/account'/change/index
inline constexpr uint32_t HD_PURPOSE = 44;
inline constexpr uint32_t HD_COIN_TYPE = 9555;

/// HD key chain (external = 0, internal/change = 1).
enum class HDChain : uint32_t {
    EXTERNAL = 0,
    INTERNAL = 1,
};

/// HD wallet state.
struct HDState {
    std::string mnemonic;                 ///< 24-word BIP39 mnemonic
    std::array<uint8_t, 64> seed{};       ///< 512-bit BIP39 seed
    crypto::ExtKey master_key;            ///< BIP32 master key
    uint32_t account = 0;                 ///< Current account index
    uint32_t next_external_index = 0;     ///< Next external (receive) key index
    uint32_t next_internal_index = 0;     ///< Next internal (change) key index

    void wipe();
};

/// HDKeyManager: BIP32/44 HD key derivation for ResonanceNet.
/// Derives keys along m/44'/9555'/account'/chain/index.
class HDKeyManager {
public:
    HDKeyManager() = default;
    ~HDKeyManager();

    HDKeyManager(const HDKeyManager&) = delete;
    HDKeyManager& operator=(const HDKeyManager&) = delete;

    /// Create a new HD wallet from a fresh 24-word mnemonic.
    Result<std::string> create(const std::string& passphrase = "");

    /// Restore from an existing mnemonic.
    Result<void> restore(const std::string& mnemonic,
                         const std::string& passphrase = "");

    /// Check if HD wallet is initialized.
    bool is_initialized() const;

    /// Get the mnemonic (for backup display).
    Result<std::string> get_mnemonic() const;

    /// Derive the next external (receive) key.
    Result<WalletKey> derive_next_external();

    /// Derive the next internal (change) key.
    Result<WalletKey> derive_next_internal();

    /// Derive a specific key by chain and index.
    Result<WalletKey> derive_key(HDChain chain, uint32_t index) const;

    /// Get the current account index.
    uint32_t get_account() const;

    /// Get the next external key index.
    uint32_t get_next_external_index() const;

    /// Get the next internal key index.
    uint32_t get_next_internal_index() const;

    /// Get the HD state (for serialization/backup).
    const HDState& state() const { return state_; }

    /// Set HD state (for loading from DB).
    void set_state(HDState s);

private:
    mutable core::Mutex mutex_;
    HDState state_;
    bool initialized_ = false;

    /// Internal: derive key at path m/44'/9555'/account'/chain/index.
    Result<WalletKey> derive_at(uint32_t chain, uint32_t index) const;
};

}  // namespace rnet::wallet
