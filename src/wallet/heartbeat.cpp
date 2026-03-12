#include "wallet/heartbeat.h"

#include "core/logging.h"
#include "primitives/amount.h"
#include "primitives/txout.h"
#include "wallet/sign_tx.h"

#include <algorithm>

namespace rnet::wallet {

Result<primitives::CMutableTransaction> HeartbeatCreator::create_heartbeat_tx(
    const std::vector<WalletCoin>& available_coins,
    const uint160& change_hash,
    int64_t fee_per_kvb) {

    if (available_coins.empty()) {
        return Result<primitives::CMutableTransaction>::err("no coins available");
    }

    // Select the coin with the highest value (maximize heartbeat impact)
    const WalletCoin* best = nullptr;
    for (const auto& coin : available_coins) {
        if (coin.is_spent) continue;
        if (!best || coin.txout.value > best->txout.value) {
            best = &coin;
        }
    }

    if (!best) {
        return Result<primitives::CMutableTransaction>::err("no unspent coins");
    }

    // Estimate fee (1 input + 1 output)
    // P2WPKH input ~68 vbytes, output ~43 vbytes, overhead ~11 vbytes
    size_t estimated_vsize = 68 + 43 + 11;
    int64_t fee = fee_per_kvb * static_cast<int64_t>(estimated_vsize) / 1000;
    if (fee < 1) fee = 1;

    int64_t output_value = best->txout.value - fee;
    if (output_value <= 0) {
        return Result<primitives::CMutableTransaction>::err(
            "coin value too small for heartbeat fee");
    }

    // Build heartbeat transaction
    primitives::CMutableTransaction mtx;
    mtx.version = primitives::TX_VERSION_HEARTBEAT;  // version = 3

    // Input: spend the selected coin
    primitives::CTxIn txin(best->outpoint);
    txin.sequence = primitives::SEQUENCE_FINAL;
    mtx.vin.push_back(std::move(txin));

    // Output: send back to self (same pubkey hash)
    auto script = primitives::make_p2wpkh_script(change_hash.data());
    mtx.vout.emplace_back(output_value, std::move(script));

    return Result<primitives::CMutableTransaction>::ok(std::move(mtx));
}

Result<void> HeartbeatCreator::sign_heartbeat_tx(
    const KeyStore& keys,
    primitives::CMutableTransaction& tx,
    const std::vector<WalletCoin>& spent_coins) {

    int signed_count = sign_wallet_transaction(keys, tx, spent_coins);
    if (signed_count == 0) {
        return Result<void>::err("failed to sign heartbeat transaction");
    }
    if (signed_count != static_cast<int>(tx.vin.size())) {
        return Result<void>::err("partially signed heartbeat transaction");
    }
    return Result<void>::ok();
}

bool HeartbeatCreator::is_valid_heartbeat(
    const primitives::CMutableTransaction& tx) {
    // Must be version 3
    if (tx.version != primitives::TX_VERSION_HEARTBEAT) {
        return false;
    }
    // Must have at least 1 input and 1 output
    if (tx.vin.empty() || tx.vout.empty()) {
        return false;
    }
    // All outputs must be P2WPKH (self-spend)
    for (const auto& out : tx.vout) {
        if (!out.is_p2wpkh()) {
            return false;
        }
    }
    return true;
}

}  // namespace rnet::wallet
