#include "wallet/recovery.h"

#include "core/logging.h"

namespace rnet::wallet {

Result<void> RecoveryManager::set_policy(script::RecoveryType type,
                                         const script::RecoveryPolicy& policy) {
    LOCK(mutex_);

    // Validate policy
    if (type == script::RecoveryType::HEARTBEAT) {
        if (!std::holds_alternative<script::HeartbeatPolicy>(policy)) {
            return Result<void>::err("policy type mismatch: expected HeartbeatPolicy");
        }
        const auto& hb = std::get<script::HeartbeatPolicy>(policy);
        if (hb.interval == 0) {
            return Result<void>::err("heartbeat interval must be > 0");
        }
        if (hb.recovery_pubkey_hash.empty()) {
            return Result<void>::err("recovery pubkey hash is required");
        }
    } else if (type == script::RecoveryType::SOCIAL) {
        if (!std::holds_alternative<script::SocialPolicy>(policy)) {
            return Result<void>::err("policy type mismatch: expected SocialPolicy");
        }
        const auto& sp = std::get<script::SocialPolicy>(policy);
        if (sp.threshold == 0 || sp.guardian_pubkeys.empty()) {
            return Result<void>::err("social recovery requires guardians");
        }
        if (sp.threshold > sp.guardian_pubkeys.size()) {
            return Result<void>::err("threshold exceeds guardian count");
        }
    } else if (type == script::RecoveryType::EMISSION) {
        if (!std::holds_alternative<script::EmissionPolicy>(policy)) {
            return Result<void>::err("policy type mismatch: expected EmissionPolicy");
        }
        const auto& ep = std::get<script::EmissionPolicy>(policy);
        if (ep.inactivity_period == 0) {
            return Result<void>::err("emission inactivity period must be > 0");
        }
    }

    type_ = type;
    policy_ = policy;
    has_policy_ = true;

    LogPrint(WALLET, "recovery policy set: type=%d",
             static_cast<int>(type));
    return Result<void>::ok();
}

bool RecoveryManager::has_policy() const {
    LOCK(mutex_);
    return has_policy_;
}

script::RecoveryType RecoveryManager::get_type() const {
    LOCK(mutex_);
    return type_;
}

const script::RecoveryPolicy& RecoveryManager::get_policy() const {
    LOCK(mutex_);
    return policy_;
}

Result<uint64_t> RecoveryManager::get_heartbeat_interval() const {
    LOCK(mutex_);
    if (!has_policy_ || type_ != script::RecoveryType::HEARTBEAT) {
        return Result<uint64_t>::err("no heartbeat policy set");
    }
    const auto& hb = std::get<script::HeartbeatPolicy>(policy_);
    return Result<uint64_t>::ok(hb.interval);
}

bool RecoveryManager::heartbeat_due(uint64_t blocks_since_last) const {
    LOCK(mutex_);
    if (!has_policy_ || type_ != script::RecoveryType::HEARTBEAT) {
        return false;
    }
    const auto& hb = std::get<script::HeartbeatPolicy>(policy_);
    // Due when 80% of interval has passed (early warning)
    return blocks_since_last >= (hb.interval * 80 / 100);
}

uint64_t RecoveryManager::blocks_until_due(uint64_t blocks_since_last) const {
    LOCK(mutex_);
    if (!has_policy_ || type_ != script::RecoveryType::HEARTBEAT) {
        return 0;
    }
    const auto& hb = std::get<script::HeartbeatPolicy>(policy_);
    uint64_t threshold = hb.interval * 80 / 100;
    if (blocks_since_last >= threshold) {
        return 0;
    }
    return threshold - blocks_since_last;
}

Result<script::CScript> RecoveryManager::build_script(
    const std::vector<uint8_t>& owner_pubkey_hash) const {
    LOCK(mutex_);
    if (!has_policy_) {
        return Result<script::CScript>::err("no recovery policy set");
    }
    auto script = script::build_recovery_script(owner_pubkey_hash, type_, policy_);
    return Result<script::CScript>::ok(std::move(script));
}

}  // namespace rnet::wallet
