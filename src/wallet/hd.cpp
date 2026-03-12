#include "wallet/hd.h"

#include "core/logging.h"
#include "crypto/hash.h"

#include <chrono>

namespace rnet::wallet {

void HDState::wipe() {
    crypto::secure_wipe(seed.data(), seed.size());
    mnemonic.clear();
    master_key = crypto::ExtKey{};
}

HDKeyManager::~HDKeyManager() {
    LOCK(mutex_);
    state_.wipe();
}

Result<std::string> HDKeyManager::create(const std::string& passphrase) {
    LOCK(mutex_);
    if (initialized_) {
        return Result<std::string>::err("HD wallet already initialized");
    }

    // Generate 24-word mnemonic (256 bits entropy)
    auto mnemonic_result = crypto::generate_mnemonic(24);
    if (!mnemonic_result) {
        return Result<std::string>::err("mnemonic generation failed: " + mnemonic_result.error());
    }

    state_.mnemonic = mnemonic_result.value();

    // Derive seed from mnemonic + passphrase
    auto seed_result = crypto::mnemonic_to_seed(state_.mnemonic, passphrase);
    if (!seed_result) {
        return Result<std::string>::err("seed derivation failed: " + seed_result.error());
    }
    state_.seed = seed_result.value();

    // Derive master key
    auto master_result = crypto::master_key_from_seed(
        std::span<const uint8_t>(state_.seed.data(), state_.seed.size()));
    if (!master_result) {
        return Result<std::string>::err("master key derivation failed: " + master_result.error());
    }
    state_.master_key = master_result.value();

    state_.account = 0;
    state_.next_external_index = 0;
    state_.next_internal_index = 0;
    initialized_ = true;

    LogPrint(WALLET, "HD wallet created with 24-word mnemonic");
    return Result<std::string>::ok(state_.mnemonic);
}

Result<void> HDKeyManager::restore(const std::string& mnemonic,
                                   const std::string& passphrase) {
    LOCK(mutex_);
    if (initialized_) {
        return Result<void>::err("HD wallet already initialized");
    }

    if (!crypto::validate_mnemonic(mnemonic)) {
        return Result<void>::err("invalid mnemonic");
    }

    state_.mnemonic = mnemonic;

    auto seed_result = crypto::mnemonic_to_seed(mnemonic, passphrase);
    if (!seed_result) {
        return Result<void>::err("seed derivation failed: " + seed_result.error());
    }
    state_.seed = seed_result.value();

    auto master_result = crypto::master_key_from_seed(
        std::span<const uint8_t>(state_.seed.data(), state_.seed.size()));
    if (!master_result) {
        return Result<void>::err("master key derivation failed: " + master_result.error());
    }
    state_.master_key = master_result.value();

    state_.account = 0;
    state_.next_external_index = 0;
    state_.next_internal_index = 0;
    initialized_ = true;

    LogPrint(WALLET, "HD wallet restored from mnemonic");
    return Result<void>::ok();
}

bool HDKeyManager::is_initialized() const {
    LOCK(mutex_);
    return initialized_;
}

Result<std::string> HDKeyManager::get_mnemonic() const {
    LOCK(mutex_);
    if (!initialized_) {
        return Result<std::string>::err("HD wallet not initialized");
    }
    return Result<std::string>::ok(state_.mnemonic);
}

Result<WalletKey> HDKeyManager::derive_next_external() {
    LOCK(mutex_);
    if (!initialized_) {
        return Result<WalletKey>::err("HD wallet not initialized");
    }
    uint32_t idx = state_.next_external_index;
    auto result = derive_at(static_cast<uint32_t>(HDChain::EXTERNAL), idx);
    if (result) {
        state_.next_external_index++;
    }
    return result;
}

Result<WalletKey> HDKeyManager::derive_next_internal() {
    LOCK(mutex_);
    if (!initialized_) {
        return Result<WalletKey>::err("HD wallet not initialized");
    }
    uint32_t idx = state_.next_internal_index;
    auto result = derive_at(static_cast<uint32_t>(HDChain::INTERNAL), idx);
    if (result) {
        state_.next_internal_index++;
    }
    return result;
}

Result<WalletKey> HDKeyManager::derive_key(HDChain chain, uint32_t index) const {
    LOCK(mutex_);
    if (!initialized_) {
        return Result<WalletKey>::err("HD wallet not initialized");
    }
    return derive_at(static_cast<uint32_t>(chain), index);
}

uint32_t HDKeyManager::get_account() const {
    LOCK(mutex_);
    return state_.account;
}

uint32_t HDKeyManager::get_next_external_index() const {
    LOCK(mutex_);
    return state_.next_external_index;
}

uint32_t HDKeyManager::get_next_internal_index() const {
    LOCK(mutex_);
    return state_.next_internal_index;
}

void HDKeyManager::set_state(HDState s) {
    LOCK(mutex_);
    state_ = std::move(s);
    initialized_ = true;
}

Result<WalletKey> HDKeyManager::derive_at(uint32_t chain, uint32_t index) const {
    // Path: m/44'/9555'/account'/chain/index
    auto key_result = crypto::derive_rnet_key(
        state_.master_key, state_.account, chain, index);
    if (!key_result) {
        return Result<WalletKey>::err("key derivation failed: " + key_result.error());
    }
    auto& ext_key = key_result.value();

    // Get public key from derived private key
    auto pubkey_result = ext_key.get_pubkey();
    if (!pubkey_result) {
        return Result<WalletKey>::err("pubkey derivation failed: " + pubkey_result.error());
    }

    // Build Ed25519 keypair from derived key
    auto kp_result = crypto::ed25519_from_seed(
        std::span<const uint8_t>(ext_key.key.data(), ext_key.key.size()));
    if (!kp_result) {
        return Result<WalletKey>::err("ed25519 keypair from seed failed: " + kp_result.error());
    }

    WalletKey wk;
    wk.secret = kp_result.value().secret;
    wk.pubkey = kp_result.value().public_key;
    wk.pubkey_hash = compute_pubkey_hash(wk.pubkey);
    wk.creation_time = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    wk.is_hd = true;
    wk.hd_index = index;
    wk.hd_change = chain;

    return Result<WalletKey>::ok(std::move(wk));
}

}  // namespace rnet::wallet
