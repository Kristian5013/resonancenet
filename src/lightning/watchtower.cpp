#include "lightning/watchtower.h"

#include "core/logging.h"
#include "crypto/keccak.h"

namespace rnet::lightning {

uint256 Watchtower::compute_hint(const uint256& txid) {
    // Use first 16 bytes of keccak256d(txid) as the hint
    auto hash = crypto::keccak256d(txid.span());
    // Zero out the last 16 bytes for a compact match
    uint256 hint;
    std::memcpy(hint.data(), hash.data(), 16);
    return hint;
}

Result<void> Watchtower::watch_channel(
    const ChannelId& channel_id,
    const crypto::Ed25519PublicKey& our_pubkey,
    const crypto::Ed25519PublicKey& their_pubkey) {

    LOCK(mutex_);

    if (watched_.count(channel_id)) {
        return Result<void>::err("Channel already being watched");
    }

    WatchedChannel wc;
    wc.channel_id = channel_id;
    wc.our_pubkey = our_pubkey;
    wc.their_pubkey = their_pubkey;
    wc.latest_commitment = 0;

    watched_[channel_id] = std::move(wc);

    LogPrint(LIGHTNING, "Watchtower: watching channel %s",
             channel_id.to_hex().c_str());
    return Result<void>::ok();
}

Result<void> Watchtower::unwatch_channel(const ChannelId& channel_id) {
    LOCK(mutex_);

    auto it = watched_.find(channel_id);
    if (it == watched_.end()) {
        return Result<void>::err("Channel not being watched");
    }

    watched_.erase(it);

    LogPrint(LIGHTNING, "Watchtower: unwatched channel %s",
             channel_id.to_hex().c_str());
    return Result<void>::ok();
}

Result<void> Watchtower::add_revocation(
    const ChannelId& channel_id,
    uint64_t commitment_number,
    const uint256& revocation_secret,
    const primitives::CMutableTransaction& justice_tx) {

    LOCK(mutex_);

    auto it = watched_.find(channel_id);
    if (it == watched_.end()) {
        return Result<void>::err("Channel not being watched");
    }

    auto& wc = it->second;

    BreachRemedy remedy;
    remedy.channel_id = channel_id;
    remedy.commitment_number = commitment_number;
    remedy.revocation_secret = revocation_secret;
    remedy.justice_tx = justice_tx;

    // Compute the hint from the revocation secret
    // The hint is derived from the commitment transaction's txid
    // which can be reconstructed from the commitment number and keys
    uint256 hint = compute_hint(revocation_secret);
    wc.remedies[hint] = std::move(remedy);

    if (commitment_number > wc.latest_commitment) {
        wc.latest_commitment = commitment_number;
    }

    LogPrint(LIGHTNING, "Watchtower: added revocation %llu for channel %s",
             commitment_number, channel_id.to_hex().c_str());
    return Result<void>::ok();
}

uint32_t Watchtower::process_block(
    const std::vector<primitives::CTransaction>& transactions,
    uint32_t block_height) {

    LOCK(mutex_);

    uint32_t breaches = 0;

    for (const auto& tx : transactions) {
        uint256 txid = tx.txid();
        uint256 hint = compute_hint(txid);

        // Check each watched channel for this hint
        for (auto& [channel_id, wc] : watched_) {
            auto rit = wc.remedies.find(hint);
            if (rit != wc.remedies.end()) {
                // Breach detected!
                LogPrint(LIGHTNING,
                    "BREACH DETECTED: channel %s, commitment %llu, block %u",
                    channel_id.to_hex().c_str(),
                    rit->second.commitment_number,
                    block_height);

                if (breach_callback_) {
                    breach_callback_(channel_id, rit->second.justice_tx);
                }
                ++breaches;
            }
        }
    }

    return breaches;
}

void Watchtower::set_breach_callback(BreachCallback callback) {
    LOCK(mutex_);
    breach_callback_ = std::move(callback);
}

Result<BreachRemedy> Watchtower::check_transaction(
    const uint256& txid,
    const ChannelId& channel_id) const {

    LOCK(mutex_);

    auto wit = watched_.find(channel_id);
    if (wit == watched_.end()) {
        return Result<BreachRemedy>::err("Channel not being watched");
    }

    uint256 hint = compute_hint(txid);
    auto rit = wit->second.remedies.find(hint);
    if (rit == wit->second.remedies.end()) {
        return Result<BreachRemedy>::err("No breach detected for this transaction");
    }

    return Result<BreachRemedy>::ok(rit->second);
}

size_t Watchtower::watched_channel_count() const {
    LOCK(mutex_);
    return watched_.size();
}

size_t Watchtower::total_revocation_count() const {
    LOCK(mutex_);
    size_t total = 0;
    for (const auto& [_, wc] : watched_) {
        total += wc.remedies.size();
    }
    return total;
}

const WatchedChannel* Watchtower::get_watched(const ChannelId& channel_id) const {
    LOCK(mutex_);
    auto it = watched_.find(channel_id);
    return it != watched_.end() ? &it->second : nullptr;
}

void Watchtower::clear() {
    LOCK(mutex_);
    watched_.clear();
}

}  // namespace rnet::lightning
