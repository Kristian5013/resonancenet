// Copyright (c) 2025-2026 The ResonanceNet Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://opensource.org/licenses/MIT.

#include "wallet/notify.h"

#include "core/logging.h"
#include "primitives/amount.h"

namespace rnet::wallet {

// ===========================================================================
//  WalletNotifier -- handle-based observer for wallet events
// ===========================================================================
//
//  Consumers register callbacks and receive a numeric handle for later
//  unregistration.  Events cover the full wallet lifecycle: incoming /
//  outgoing txs, confirmations, block connections, balance changes, and
//  heartbeat deadlines.

// ---------------------------------------------------------------------------
// register_callback / unregister_callback
// ---------------------------------------------------------------------------

size_t WalletNotifier::register_callback(WalletNotifyCallback cb) {
    LOCK(mutex_);
    size_t handle = next_handle_++;
    callbacks_.emplace_back(handle, std::move(cb));
    return handle;
}

void WalletNotifier::unregister_callback(size_t handle) {
    LOCK(mutex_);
    callbacks_.erase(
        std::remove_if(callbacks_.begin(), callbacks_.end(),
                        [handle](const auto& p) { return p.first == handle; }),
        callbacks_.end());
}

// ---------------------------------------------------------------------------
// notify -- dispatch an event to all registered callbacks
// ---------------------------------------------------------------------------

void WalletNotifier::notify(const WalletNotifyEvent& event) const {
    LOCK(mutex_);
    for (const auto& [_, cb] : callbacks_) {
        cb(event);
    }
}

// ---------------------------------------------------------------------------
// Typed notification helpers
// ---------------------------------------------------------------------------

void WalletNotifier::notify_tx_received(const uint256& txid, int64_t amount,
                                        const std::string& address) {
    WalletNotifyEvent event;
    event.type = WalletNotifyType::TX_RECEIVED;
    event.txid = txid;
    event.amount = amount;
    event.address = address;
    event.message = "Received " + primitives::FormatMoney(amount) + " RNT";
    notify(event);
}

void WalletNotifier::notify_tx_sent(const uint256& txid, int64_t amount) {
    WalletNotifyEvent event;
    event.type = WalletNotifyType::TX_SENT;
    event.txid = txid;
    event.amount = amount;
    event.message = "Sent " + primitives::FormatMoney(amount) + " RNT";
    notify(event);
}

void WalletNotifier::notify_tx_confirmed(const uint256& txid, int32_t height) {
    WalletNotifyEvent event;
    event.type = WalletNotifyType::TX_CONFIRMED;
    event.txid = txid;
    event.height = height;
    event.message = "Transaction confirmed at height " + std::to_string(height);
    notify(event);
}

void WalletNotifier::notify_block_connected(const uint256& block_hash,
                                            int32_t height) {
    WalletNotifyEvent event;
    event.type = WalletNotifyType::BLOCK_CONNECTED;
    event.block_hash = block_hash;
    event.height = height;
    event.message = "Block " + std::to_string(height) + " connected";
    notify(event);
}

void WalletNotifier::notify_balance_changed(int64_t new_balance) {
    WalletNotifyEvent event;
    event.type = WalletNotifyType::BALANCE_CHANGED;
    event.amount = new_balance;
    event.message = "Balance: " + primitives::FormatMoney(new_balance) + " RNT";
    notify(event);
}

void WalletNotifier::notify_heartbeat_due(uint64_t blocks_remaining) {
    WalletNotifyEvent event;
    event.type = WalletNotifyType::HEARTBEAT_DUE;
    event.message = "Heartbeat due in " + std::to_string(blocks_remaining) + " blocks";
    notify(event);
}

// ---------------------------------------------------------------------------
// clear -- remove all registered callbacks
// ---------------------------------------------------------------------------

void WalletNotifier::clear() {
    LOCK(mutex_);
    callbacks_.clear();
}

} // namespace rnet::wallet
