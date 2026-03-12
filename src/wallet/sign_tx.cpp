#include "wallet/sign_tx.h"

#include "core/logging.h"
#include "script/script.h"
#include "script/standard.h"

namespace rnet::wallet {

WalletSigningProvider::WalletSigningProvider(const KeyStore& keys)
    : keys_(keys) {}

bool WalletSigningProvider::get_ed25519_key(
    const std::vector<uint8_t>& pubkey_hash,
    std::vector<uint8_t>& secret_out,
    std::vector<uint8_t>& pubkey_out) const {

    if (pubkey_hash.size() != 20) return false;

    uint160 hash(std::span<const uint8_t>(pubkey_hash.data(), pubkey_hash.size()));
    return keys_.get_signing_key(hash, secret_out, pubkey_out);
}

int sign_wallet_transaction(
    const KeyStore& keys,
    primitives::CMutableTransaction& mtx,
    const std::vector<WalletCoin>& spent_coins) {

    if (mtx.vin.size() != spent_coins.size()) {
        LogPrint(WALLET, "sign_wallet_transaction: input count mismatch");
        return 0;
    }

    WalletSigningProvider provider(keys);

    std::vector<script::CScript> prev_scripts;
    std::vector<int64_t> prev_amounts;

    prev_scripts.reserve(spent_coins.size());
    prev_amounts.reserve(spent_coins.size());

    for (const auto& coin : spent_coins) {
        prev_scripts.emplace_back(
            coin.txout.script_pub_key.begin(),
            coin.txout.script_pub_key.end());
        prev_amounts.push_back(coin.txout.value);
    }

    return script::sign_transaction(provider, mtx, prev_scripts, prev_amounts);
}

bool is_fully_signed(const primitives::CMutableTransaction& tx) {
    for (const auto& txin : tx.vin) {
        // For P2WPKH, witness must have 2 items (sig + pubkey)
        if (txin.witness.stack.size() < 2) {
            // Check if it has a non-empty scriptSig (legacy)
            if (txin.script_sig.empty()) {
                return false;
            }
        }
    }
    return true;
}

}  // namespace rnet::wallet
