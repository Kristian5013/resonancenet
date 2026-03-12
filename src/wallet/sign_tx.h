#pragma once

#include <cstdint>
#include <vector>

#include "core/error.h"
#include "primitives/transaction.h"
#include "script/sign.h"
#include "wallet/coins.h"
#include "wallet/keys.h"

namespace rnet::wallet {

/// WalletSigningProvider: implements SigningProvider using the wallet's KeyStore.
class WalletSigningProvider : public script::SigningProvider {
public:
    explicit WalletSigningProvider(const KeyStore& keys);

    bool get_ed25519_key(
        const std::vector<uint8_t>& pubkey_hash,
        std::vector<uint8_t>& secret_out,
        std::vector<uint8_t>& pubkey_out) const override;

private:
    const KeyStore& keys_;
};

/// Sign a transaction using wallet keys.
/// @param keys     The wallet's key store.
/// @param mtx      Mutable transaction to sign (modified in place).
/// @param coins    The coins being spent (for amounts and scripts).
/// @return Number of inputs successfully signed.
int sign_wallet_transaction(
    const KeyStore& keys,
    primitives::CMutableTransaction& mtx,
    const std::vector<WalletCoin>& spent_coins);

/// Check if a transaction is fully signed.
bool is_fully_signed(const primitives::CMutableTransaction& tx);

}  // namespace rnet::wallet
