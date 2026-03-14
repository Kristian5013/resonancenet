// Copyright (c) 2024-present ResonanceNet developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://opensource.org/licenses/MIT.

#include "lightning/channel.h"

#include "core/logging.h"
#include "core/random.h"
#include "crypto/keccak.h"

#include <cstring>

// ---------------------------------------------------------------------------
// Design note — Channel state machine lifecycle
//
//   PREOPENING  ──>  FUNDING_CREATED  ──>  FUNDING_BROADCAST
//       |                                        |
//       v                                        v
//   (rejected)                            FUNDING_LOCKED
//                                                |
//                                                v
//                                             NORMAL
//                                           /        \
//                                     SHUTDOWN    FORCE_CLOSING
//                                          \        /
//                                           CLOSED
//
// Every public method acquires mutex_ before touching mutable state.
// State transitions are enforced by check_state() / check_state_any().
// ---------------------------------------------------------------------------

namespace rnet::lightning {

// ---------------------------------------------------------------------------
// close_type_name
// ---------------------------------------------------------------------------

std::string_view close_type_name(CloseType type) {
    switch (type) {
        case CloseType::COOPERATIVE:  return "COOPERATIVE";
        case CloseType::FORCE_LOCAL:  return "FORCE_LOCAL";
        case CloseType::FORCE_REMOTE: return "FORCE_REMOTE";
        case CloseType::BREACH:       return "BREACH";
        default:                      return "UNKNOWN";
    }
}

// ---------------------------------------------------------------------------
// LightningChannel::create_outbound
// ---------------------------------------------------------------------------

Result<LightningChannel> LightningChannel::create_outbound(
    const crypto::Ed25519KeyPair& local_keys,
    const crypto::Ed25519PublicKey& remote_node_id,
    int64_t capacity,
    int64_t push_amount,
    const ChannelConfig& config) {

    // 1. Validate capacity bounds.
    if (capacity < MIN_CHANNEL_CAPACITY) {
        return Result<LightningChannel>::err(
            "Channel capacity below minimum: " + std::to_string(capacity));
    }
    if (capacity > MAX_CHANNEL_CAPACITY) {
        return Result<LightningChannel>::err(
            "Channel capacity exceeds maximum: " + std::to_string(capacity));
    }

    // 2. Validate push amount.
    if (push_amount < 0 || push_amount > capacity) {
        return Result<LightningChannel>::err(
            "Invalid push amount: " + std::to_string(push_amount));
    }

    // 3. Populate channel fields.
    LightningChannel ch;
    ch.state_ = ChannelState::PREOPENING;
    ch.is_funder_ = true;
    ch.temp_channel_id_ = core::get_rand_hash();
    ch.capacity_ = capacity;
    ch.balance_.local = capacity - push_amount;
    ch.balance_.remote = push_amount;
    ch.local_keypair_ = local_keys;
    ch.remote_node_id_ = remote_node_id;

    // 4. Set local config with 1 % reserve.
    ch.local_state_.config = config;
    ch.local_state_.config.channel_reserve = capacity / 100;
    ch.local_state_.keys.funding_pubkey = local_keys.public_key;

    LogPrint(LIGHTNING, "Created outbound channel %s, capacity=%lld",
             ch.temp_channel_id_.to_hex().c_str(), capacity);

    return Result<LightningChannel>::ok(std::move(ch));
}

// ---------------------------------------------------------------------------
// LightningChannel::create_inbound
// ---------------------------------------------------------------------------

Result<LightningChannel> LightningChannel::create_inbound(
    const crypto::Ed25519KeyPair& local_keys,
    const crypto::Ed25519PublicKey& remote_node_id,
    int64_t capacity,
    int64_t push_amount,
    const ChannelConfig& local_config,
    const ChannelConfig& remote_config) {

    // 1. Validate capacity bounds.
    if (capacity < MIN_CHANNEL_CAPACITY) {
        return Result<LightningChannel>::err(
            "Channel capacity below minimum: " + std::to_string(capacity));
    }
    if (capacity > MAX_CHANNEL_CAPACITY) {
        return Result<LightningChannel>::err(
            "Channel capacity exceeds maximum: " + std::to_string(capacity));
    }

    // 2. Populate channel fields (we receive the push amount).
    LightningChannel ch;
    ch.state_ = ChannelState::PREOPENING;
    ch.is_funder_ = false;
    ch.temp_channel_id_ = core::get_rand_hash();
    ch.capacity_ = capacity;
    ch.balance_.local = push_amount;
    ch.balance_.remote = capacity - push_amount;
    ch.local_keypair_ = local_keys;
    ch.remote_node_id_ = remote_node_id;

    // 3. Store both sides' configs.
    ch.local_state_.config = local_config;
    ch.local_state_.config.channel_reserve = capacity / 100;
    ch.local_state_.keys.funding_pubkey = local_keys.public_key;
    ch.remote_state_.config = remote_config;

    LogPrint(LIGHTNING, "Created inbound channel %s, capacity=%lld",
             ch.temp_channel_id_.to_hex().c_str(), capacity);

    return Result<LightningChannel>::ok(std::move(ch));
}

// ---------------------------------------------------------------------------
// LightningChannel::LightningChannel  (move constructor)
// ---------------------------------------------------------------------------

LightningChannel::LightningChannel(LightningChannel&& other) noexcept {
    LOCK(other.mutex_);
    state_ = other.state_;
    is_funder_ = other.is_funder_;
    temp_channel_id_ = other.temp_channel_id_;
    channel_id_ = other.channel_id_;
    funding_outpoint_ = other.funding_outpoint_;
    capacity_ = other.capacity_;
    balance_ = other.balance_;
    local_state_ = std::move(other.local_state_);
    remote_state_ = std::move(other.remote_state_);
    htlcs_ = std::move(other.htlcs_);
    local_keypair_ = other.local_keypair_;
    remote_node_id_ = other.remote_node_id_;
    close_type_ = other.close_type_;
    closing_txid_ = other.closing_txid_;
    confirmation_height_ = other.confirmation_height_;
}

// ---------------------------------------------------------------------------
// LightningChannel::operator=  (move assignment)
// ---------------------------------------------------------------------------

LightningChannel& LightningChannel::operator=(LightningChannel&& other) noexcept {
    if (this != &other) {
        // 1. Lock both mutexes in consistent order to prevent deadlock.
        LOCK2(mutex_, other.mutex_);
        state_ = other.state_;
        is_funder_ = other.is_funder_;
        temp_channel_id_ = other.temp_channel_id_;
        channel_id_ = other.channel_id_;
        funding_outpoint_ = other.funding_outpoint_;
        capacity_ = other.capacity_;
        balance_ = other.balance_;
        local_state_ = std::move(other.local_state_);
        remote_state_ = std::move(other.remote_state_);
        htlcs_ = std::move(other.htlcs_);
        local_keypair_ = other.local_keypair_;
        remote_node_id_ = other.remote_node_id_;
        close_type_ = other.close_type_;
        closing_txid_ = other.closing_txid_;
        confirmation_height_ = other.confirmation_height_;
    }
    return *this;
}

// ---------------------------------------------------------------------------
// LightningChannel::state
// ---------------------------------------------------------------------------

ChannelState LightningChannel::state() const {
    LOCK(mutex_);
    return state_;
}

// ---------------------------------------------------------------------------
// LightningChannel::channel_id
// ---------------------------------------------------------------------------

ChannelId LightningChannel::channel_id() const {
    LOCK(mutex_);
    return channel_id_;
}

// ---------------------------------------------------------------------------
// LightningChannel::temp_channel_id
// ---------------------------------------------------------------------------

TempChannelId LightningChannel::temp_channel_id() const {
    LOCK(mutex_);
    return temp_channel_id_;
}

// ---------------------------------------------------------------------------
// LightningChannel::capacity
// ---------------------------------------------------------------------------

int64_t LightningChannel::capacity() const {
    LOCK(mutex_);
    return capacity_;
}

// ---------------------------------------------------------------------------
// LightningChannel::balance
// ---------------------------------------------------------------------------

ChannelBalance LightningChannel::balance() const {
    LOCK(mutex_);
    return balance_;
}

// ---------------------------------------------------------------------------
// LightningChannel::is_funder
// ---------------------------------------------------------------------------

bool LightningChannel::is_funder() const {
    LOCK(mutex_);
    return is_funder_;
}

// ---------------------------------------------------------------------------
// LightningChannel::local_node_id
// ---------------------------------------------------------------------------

const crypto::Ed25519PublicKey& LightningChannel::local_node_id() const {
    return local_keypair_.public_key;
}

// ---------------------------------------------------------------------------
// LightningChannel::remote_node_id
// ---------------------------------------------------------------------------

const crypto::Ed25519PublicKey& LightningChannel::remote_node_id() const {
    return remote_node_id_;
}

// ---------------------------------------------------------------------------
// LightningChannel::local_commitment_number
// ---------------------------------------------------------------------------

uint64_t LightningChannel::local_commitment_number() const {
    LOCK(mutex_);
    return local_state_.next_commitment_number;
}

// ---------------------------------------------------------------------------
// LightningChannel::remote_commitment_number
// ---------------------------------------------------------------------------

uint64_t LightningChannel::remote_commitment_number() const {
    LOCK(mutex_);
    return remote_state_.next_commitment_number;
}

// ---------------------------------------------------------------------------
// LightningChannel::check_state
// ---------------------------------------------------------------------------

Result<void> LightningChannel::check_state(ChannelState expected) const {
    if (state_ != expected) {
        return Result<void>::err(
            "Invalid channel state: expected " +
            std::string(channel_state_name(expected)) + ", got " +
            std::string(channel_state_name(state_)));
    }
    return Result<void>::ok();
}

// ---------------------------------------------------------------------------
// LightningChannel::check_state_any
// ---------------------------------------------------------------------------

Result<void> LightningChannel::check_state_any(
    std::initializer_list<ChannelState> expected) const {
    for (auto s : expected) {
        if (state_ == s) return Result<void>::ok();
    }
    return Result<void>::err(
        "Invalid channel state: " + std::string(channel_state_name(state_)));
}

// ---------------------------------------------------------------------------
// LightningChannel::funding_created
// ---------------------------------------------------------------------------

Result<void> LightningChannel::funding_created(
    const primitives::COutPoint& funding_outpoint,
    const crypto::Ed25519Signature& /*remote_sig*/) {
    LOCK(mutex_);

    // 1. Verify we are in PREOPENING.
    auto r = check_state(ChannelState::PREOPENING);
    if (!r) return r;

    // 2. Record the funding outpoint and derive the permanent channel id.
    funding_outpoint_ = funding_outpoint;
    channel_id_ = make_channel_id(funding_outpoint);

    // 3. Advance state.
    state_ = ChannelState::FUNDING_CREATED;

    LogPrint(LIGHTNING, "Channel %s: funding created, outpoint=%s",
             channel_id_.to_hex().c_str(),
             funding_outpoint.to_string().c_str());
    return Result<void>::ok();
}

// ---------------------------------------------------------------------------
// LightningChannel::funding_broadcast
// ---------------------------------------------------------------------------

Result<void> LightningChannel::funding_broadcast(const uint256& /*funding_txid*/) {
    LOCK(mutex_);

    // 1. Verify we are in FUNDING_CREATED.
    auto r = check_state(ChannelState::FUNDING_CREATED);
    if (!r) return r;

    // 2. Advance state.
    state_ = ChannelState::FUNDING_BROADCAST;

    LogPrint(LIGHTNING, "Channel %s: funding broadcast",
             channel_id_.to_hex().c_str());
    return Result<void>::ok();
}

// ---------------------------------------------------------------------------
// LightningChannel::funding_locked
// ---------------------------------------------------------------------------

Result<void> LightningChannel::funding_locked(uint32_t confirmation_height) {
    LOCK(mutex_);

    // 1. Verify we are in FUNDING_BROADCAST.
    auto r = check_state(ChannelState::FUNDING_BROADCAST);
    if (!r) return r;

    // 2. Record confirmation height.
    confirmation_height_ = confirmation_height;
    state_ = ChannelState::FUNDING_LOCKED;

    // 3. Transition immediately to NORMAL after locking.
    state_ = ChannelState::NORMAL;

    LogPrint(LIGHTNING, "Channel %s: funding locked at height %u, now NORMAL",
             channel_id_.to_hex().c_str(), confirmation_height);
    return Result<void>::ok();
}

// ---------------------------------------------------------------------------
// LightningChannel::initiate_shutdown
// ---------------------------------------------------------------------------

Result<void> LightningChannel::initiate_shutdown() {
    LOCK(mutex_);

    // 1. Only NORMAL channels may initiate cooperative shutdown.
    auto r = check_state(ChannelState::NORMAL);
    if (!r) return r;

    // 2. Advance state.
    state_ = ChannelState::SHUTDOWN;

    LogPrint(LIGHTNING, "Channel %s: shutdown initiated",
             channel_id_.to_hex().c_str());
    return Result<void>::ok();
}

// ---------------------------------------------------------------------------
// LightningChannel::force_close
// ---------------------------------------------------------------------------

Result<CommitmentTx> LightningChannel::force_close() {
    LOCK(mutex_);

    // 1. Allow force-close from NORMAL or SHUTDOWN.
    auto r = check_state_any({ChannelState::NORMAL, ChannelState::SHUTDOWN});
    if (!r) return Result<CommitmentTx>::err(r.error());

    // 2. Build the commitment transaction.
    CommitmentTx commit_tx;
    commit_tx.commitment_number = local_state_.next_commitment_number;
    commit_tx.balance = balance_;

    primitives::CMutableTransaction mtx;
    mtx.version = 2;
    mtx.locktime = 0;

    // 3. Input: funding outpoint.
    primitives::CTxIn funding_input;
    funding_input.prevout = funding_outpoint_;
    funding_input.sequence = 0xFFFFFFFF;
    mtx.vin.push_back(std::move(funding_input));

    // 4. Output 0: to_local with CSV delay (simplified P2WPKH).
    if (balance_.local > local_state_.config.dust_limit) {
        primitives::CTxOut local_output;
        local_output.value = balance_.local;
        local_output.script_pub_key.resize(22);
        local_output.script_pub_key[0] = 0x00;
        local_output.script_pub_key[1] = 0x14;
        std::memcpy(local_output.script_pub_key.data() + 2,
                     local_keypair_.public_key.data.data(), 20);
        mtx.vout.push_back(std::move(local_output));
    }

    // 5. Output 1: to_remote (immediate P2WPKH).
    if (balance_.remote > remote_state_.config.dust_limit) {
        primitives::CTxOut remote_output;
        remote_output.value = balance_.remote;
        remote_output.script_pub_key.resize(22);
        remote_output.script_pub_key[0] = 0x00;
        remote_output.script_pub_key[1] = 0x14;
        std::memcpy(remote_output.script_pub_key.data() + 2,
                     remote_node_id_.data.data(), 20);
        mtx.vout.push_back(std::move(remote_output));
    }

    // 6. Finalize and advance state.
    commit_tx.tx = std::move(mtx);
    state_ = ChannelState::FORCE_CLOSING;

    LogPrint(LIGHTNING, "Channel %s: force-closed",
             channel_id_.to_hex().c_str());

    return Result<CommitmentTx>::ok(std::move(commit_tx));
}

// ---------------------------------------------------------------------------
// LightningChannel::mark_closed
// ---------------------------------------------------------------------------

Result<void> LightningChannel::mark_closed(CloseType type,
                                            const uint256& closing_txid) {
    LOCK(mutex_);

    // 1. Record close metadata.
    close_type_ = type;
    closing_txid_ = closing_txid;

    // 2. Terminal state.
    state_ = ChannelState::CLOSED;

    LogPrint(LIGHTNING, "Channel %s: closed (%s)",
             channel_id_.to_hex().c_str(),
             std::string(close_type_name(type)).c_str());
    return Result<void>::ok();
}

// ---------------------------------------------------------------------------
// LightningChannel::add_htlc
// ---------------------------------------------------------------------------

Result<uint64_t> LightningChannel::add_htlc(int64_t amount,
                                             const uint256& payment_hash,
                                             uint32_t cltv_expiry) {
    LOCK(mutex_);

    // 1. Require NORMAL state for HTLC operations.
    auto r = check_state(ChannelState::NORMAL);
    if (!r) return Result<uint64_t>::err(r.error());

    // 2. Build the HTLC descriptor.
    Htlc htlc;
    htlc.direction = HtlcDirection::OFFERED;
    htlc.amount = amount;
    htlc.payment_hash = payment_hash;
    htlc.cltv_expiry = cltv_expiry;

    // 3. Validate against channel limits.
    auto vr = validate_htlc(htlc, capacity_, balance_.local,
                             htlcs_.pending_count(),
                             local_state_.config.max_accepted_htlcs,
                             remote_state_.config.min_htlc_value);
    if (!vr) return Result<uint64_t>::err(vr.error());

    // 4. Insert and deduct from local balance provisionally.
    uint64_t id = htlcs_.next_id();
    auto ar = htlcs_.add(std::move(htlc));
    if (!ar) return Result<uint64_t>::err(ar.error());

    balance_.local -= amount;

    return Result<uint64_t>::ok(id);
}

// ---------------------------------------------------------------------------
// LightningChannel::fulfill_htlc
// ---------------------------------------------------------------------------

Result<void> LightningChannel::fulfill_htlc(uint64_t htlc_id,
                                             const uint256& preimage) {
    LOCK(mutex_);

    // 1. Require NORMAL state.
    auto r = check_state(ChannelState::NORMAL);
    if (!r) return r;

    // 2. Look up the HTLC.
    auto* htlc = htlcs_.find(htlc_id);
    if (!htlc) return Result<void>::err("HTLC not found");
    if (htlc->direction != HtlcDirection::RECEIVED) {
        return Result<void>::err("Can only fulfill received HTLCs");
    }

    // 3. Fulfill with preimage and credit to local balance.
    int64_t amount = htlc->amount;
    auto fr = htlcs_.fulfill(htlc_id, preimage);
    if (!fr) return fr;

    balance_.local += amount;

    return Result<void>::ok();
}

// ---------------------------------------------------------------------------
// LightningChannel::fail_htlc
// ---------------------------------------------------------------------------

Result<void> LightningChannel::fail_htlc(uint64_t htlc_id) {
    LOCK(mutex_);

    // 1. Require NORMAL state.
    auto r = check_state(ChannelState::NORMAL);
    if (!r) return r;

    // 2. Look up the HTLC.
    auto* htlc = htlcs_.find(htlc_id);
    if (!htlc) return Result<void>::err("HTLC not found");

    int64_t amount = htlc->amount;
    HtlcDirection dir = htlc->direction;

    // 3. Mark as failed.
    auto fr = htlcs_.fail(htlc_id);
    if (!fr) return fr;

    // 4. Return funds based on direction.
    if (dir == HtlcDirection::OFFERED) {
        balance_.local += amount;
    } else {
        balance_.remote += amount;
    }

    return Result<void>::ok();
}

// ---------------------------------------------------------------------------
// LightningChannel::receive_htlc
// ---------------------------------------------------------------------------

Result<uint64_t> LightningChannel::receive_htlc(int64_t amount,
                                                  const uint256& payment_hash,
                                                  uint32_t cltv_expiry) {
    LOCK(mutex_);

    // 1. Require NORMAL state.
    auto r = check_state(ChannelState::NORMAL);
    if (!r) return Result<uint64_t>::err(r.error());

    // 2. Build the HTLC descriptor.
    Htlc htlc;
    htlc.direction = HtlcDirection::RECEIVED;
    htlc.amount = amount;
    htlc.payment_hash = payment_hash;
    htlc.cltv_expiry = cltv_expiry;

    // 3. Validate against channel limits.
    auto vr = validate_htlc(htlc, capacity_, balance_.remote,
                             htlcs_.pending_count(),
                             local_state_.config.max_accepted_htlcs,
                             local_state_.config.min_htlc_value);
    if (!vr) return Result<uint64_t>::err(vr.error());

    // 4. Insert and deduct from remote balance.
    uint64_t id = htlcs_.next_id();
    auto ar = htlcs_.add(std::move(htlc));
    if (!ar) return Result<uint64_t>::err(ar.error());

    balance_.remote -= amount;

    return Result<uint64_t>::ok(id);
}

// ---------------------------------------------------------------------------
// LightningChannel::sign_local_commitment
// ---------------------------------------------------------------------------

Result<crypto::Ed25519Signature> LightningChannel::sign_local_commitment() {
    LOCK(mutex_);

    // 1. Allow signing in NORMAL or SHUTDOWN.
    auto r = check_state_any({ChannelState::NORMAL, ChannelState::SHUTDOWN});
    if (!r) return Result<crypto::Ed25519Signature>::err(r.error());

    // 2. Serialize commitment data for signing.
    core::DataStream ss;
    core::ser_write_u64(ss, local_state_.next_commitment_number);
    core::Serialize(ss, channel_id_);
    core::ser_write_i64(ss, balance_.local);
    core::ser_write_i64(ss, balance_.remote);

    // 3. Sign with local secret key.
    auto sig_result = crypto::ed25519_sign(
        local_keypair_.secret, ss.span());
    if (!sig_result) {
        return Result<crypto::Ed25519Signature>::err(sig_result.error());
    }

    return Result<crypto::Ed25519Signature>::ok(sig_result.value());
}

// ---------------------------------------------------------------------------
// LightningChannel::receive_commitment_sig
// ---------------------------------------------------------------------------

Result<void> LightningChannel::receive_commitment_sig(
    const crypto::Ed25519Signature& sig) {
    LOCK(mutex_);

    // 1. Allow in NORMAL or SHUTDOWN.
    auto r = check_state_any({ChannelState::NORMAL, ChannelState::SHUTDOWN});
    if (!r) return r;

    // 2. Reconstruct the message the remote signed.
    core::DataStream ss;
    core::ser_write_u64(ss, remote_state_.next_commitment_number);
    core::Serialize(ss, channel_id_);
    core::ser_write_i64(ss, balance_.remote);
    core::ser_write_i64(ss, balance_.local);

    // 3. Verify the signature against the remote's public key.
    if (!crypto::ed25519_verify(remote_node_id_, ss.span(), sig)) {
        return Result<void>::err("Invalid commitment signature from remote");
    }

    // 4. Advance remote commitment counter.
    remote_state_.next_commitment_number++;
    return Result<void>::ok();
}

// ---------------------------------------------------------------------------
// LightningChannel::revoke_and_ack
// ---------------------------------------------------------------------------

Result<uint256> LightningChannel::revoke_and_ack() {
    LOCK(mutex_);

    // 1. Allow in NORMAL or SHUTDOWN.
    auto r = check_state_any({ChannelState::NORMAL, ChannelState::SHUTDOWN});
    if (!r) return Result<uint256>::err(r.error());

    // 2. Generate per-commitment secret for the current commitment.
    uint256 secret = core::get_rand_hash();
    local_state_.last_per_commitment_secret = secret;

    // 3. Advance local commitment counter.
    local_state_.next_commitment_number++;

    return Result<uint256>::ok(secret);
}

// ---------------------------------------------------------------------------
// LightningChannel::receive_revocation
// ---------------------------------------------------------------------------

Result<void> LightningChannel::receive_revocation(
    const uint256& per_commitment_secret,
    const crypto::Ed25519PublicKey& next_per_commitment_point) {
    LOCK(mutex_);

    // 1. Allow in NORMAL or SHUTDOWN.
    auto r = check_state_any({ChannelState::NORMAL, ChannelState::SHUTDOWN});
    if (!r) return r;

    // 2. Store the revealed secret and next commitment point.
    remote_state_.last_per_commitment_secret = per_commitment_secret;
    remote_state_.current_per_commitment_point = next_per_commitment_point;

    // 3. Prune settled HTLCs after successful revocation.
    htlcs_.prune_settled();

    return Result<void>::ok();
}

// ---------------------------------------------------------------------------
// LightningChannel::htlc_set
// ---------------------------------------------------------------------------

const HtlcSet& LightningChannel::htlc_set() const {
    return htlcs_;
}

// ---------------------------------------------------------------------------
// LightningChannel::local_config
// ---------------------------------------------------------------------------

const ChannelConfig& LightningChannel::local_config() const {
    return local_state_.config;
}

// ---------------------------------------------------------------------------
// LightningChannel::remote_config
// ---------------------------------------------------------------------------

const ChannelConfig& LightningChannel::remote_config() const {
    return remote_state_.config;
}

// ---------------------------------------------------------------------------
// LightningChannel::funding_outpoint
// ---------------------------------------------------------------------------

const primitives::COutPoint& LightningChannel::funding_outpoint() const {
    return funding_outpoint_;
}

// ---------------------------------------------------------------------------
// LightningChannel::pending_htlc_count
// ---------------------------------------------------------------------------

uint32_t LightningChannel::pending_htlc_count() const {
    LOCK(mutex_);
    return htlcs_.pending_count();
}

// ---------------------------------------------------------------------------
// LightningChannel::to_string
// ---------------------------------------------------------------------------

std::string LightningChannel::to_string() const {
    LOCK(mutex_);
    return "Channel{id=" + channel_id_.to_hex() +
           ", state=" + std::string(channel_state_name(state_)) +
           ", capacity=" + std::to_string(capacity_) +
           ", local=" + std::to_string(balance_.local) +
           ", remote=" + std::to_string(balance_.remote) +
           ", htlcs=" + std::to_string(htlcs_.pending_count()) + "}";
}

} // namespace rnet::lightning
