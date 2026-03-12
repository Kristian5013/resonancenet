#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "core/sync.h"
#include "core/types.h"
#include "primitives/transaction.h"

namespace rnet::wallet {

/// Notification event types.
enum class WalletNotifyType {
    TX_RECEIVED,         ///< Incoming transaction detected
    TX_SENT,             ///< Outgoing transaction broadcast
    TX_CONFIRMED,        ///< Transaction confirmed in a block
    BLOCK_CONNECTED,     ///< New block connected
    BLOCK_DISCONNECTED,  ///< Block disconnected (reorg)
    BALANCE_CHANGED,     ///< Balance changed
    HEARTBEAT_DUE,       ///< Heartbeat transaction is due
    ADDRESS_USED,        ///< An address received funds
};

/// Notification event data.
struct WalletNotifyEvent {
    WalletNotifyType type;
    uint256 txid;                ///< Transaction hash (if applicable)
    uint256 block_hash;          ///< Block hash (if applicable)
    int32_t height = -1;         ///< Block height (if applicable)
    int64_t amount = 0;          ///< Amount (if applicable)
    std::string address;         ///< Address (if applicable)
    std::string message;         ///< Human-readable description
};

/// Callback type for notifications.
using WalletNotifyCallback = std::function<void(const WalletNotifyEvent&)>;

/// WalletNotifier: manages notification callbacks for wallet events.
class WalletNotifier {
public:
    WalletNotifier() = default;

    /// Register a notification callback. Returns a handle for removal.
    size_t register_callback(WalletNotifyCallback cb);

    /// Unregister a callback by handle.
    void unregister_callback(size_t handle);

    /// Fire a notification to all registered callbacks.
    void notify(const WalletNotifyEvent& event) const;

    /// Convenience: fire a transaction-received notification.
    void notify_tx_received(const uint256& txid, int64_t amount,
                            const std::string& address);

    /// Convenience: fire a transaction-sent notification.
    void notify_tx_sent(const uint256& txid, int64_t amount);

    /// Convenience: fire a transaction-confirmed notification.
    void notify_tx_confirmed(const uint256& txid, int32_t height);

    /// Convenience: fire a block-connected notification.
    void notify_block_connected(const uint256& block_hash, int32_t height);

    /// Convenience: fire a balance-changed notification.
    void notify_balance_changed(int64_t new_balance);

    /// Convenience: fire a heartbeat-due notification.
    void notify_heartbeat_due(uint64_t blocks_remaining);

    /// Clear all callbacks.
    void clear();

private:
    mutable core::Mutex mutex_;
    size_t next_handle_ = 1;
    std::vector<std::pair<size_t, WalletNotifyCallback>> callbacks_;
};

}  // namespace rnet::wallet
