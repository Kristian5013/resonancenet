#pragma once

#include <cstdint>
#include <optional>

#include "core/error.h"
#include "core/sync.h"
#include "core/types.h"
#include "script/recovery_script.h"

namespace rnet::wallet {

/// RecoveryManager: manages mandatory recovery policy for a wallet.
/// Every wallet MUST have a recovery policy set during creation.
class RecoveryManager {
public:
    RecoveryManager() = default;

    /// Set the recovery policy (MANDATORY at wallet creation time).
    Result<void> set_policy(script::RecoveryType type,
                            const script::RecoveryPolicy& policy);

    /// Check if a recovery policy is set.
    bool has_policy() const;

    /// Get the current recovery type.
    script::RecoveryType get_type() const;

    /// Get the current recovery policy.
    const script::RecoveryPolicy& get_policy() const;

    /// Get the heartbeat interval (only valid for HEARTBEAT type).
    Result<uint64_t> get_heartbeat_interval() const;

    /// Check if a heartbeat is due (blocks since last heartbeat >= threshold).
    /// @param blocks_since_last Blocks since the last heartbeat transaction.
    /// @return true if a heartbeat should be sent.
    bool heartbeat_due(uint64_t blocks_since_last) const;

    /// Get the number of blocks remaining before heartbeat is overdue.
    /// Returns 0 if already overdue.
    uint64_t blocks_until_due(uint64_t blocks_since_last) const;

    /// Build a recovery script for an owner pubkey hash.
    Result<script::CScript> build_script(
        const std::vector<uint8_t>& owner_pubkey_hash) const;

private:
    mutable core::Mutex mutex_;
    bool has_policy_ = false;
    script::RecoveryType type_ = script::RecoveryType::HEARTBEAT;
    script::RecoveryPolicy policy_;
};

}  // namespace rnet::wallet
