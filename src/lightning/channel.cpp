#include "lightning/channel.h"

#include "core/logging.h"
#include "core/random.h"
#include "crypto/keccak.h"

namespace rnet::lightning {

std::string_view close_type_name(CloseType type) {
    switch (type) {
        case CloseType::COOPERATIVE:  return "COOPERATIVE";
        case CloseType::FORCE_LOCAL:  return "FORCE_LOCAL";
        case CloseType::FORCE_REMOTE: return "FORCE_REMOTE";
        case CloseType::BREACH:       return "BREACH";
        default:                      return "UNKNOWN";
    }
}

// ── Construction ────────────────────────────────────────────────────

Result<LightningChannel> LightningChannel::create_outbound(
    const crypto::Ed25519KeyPair& local_keys,
    const crypto::Ed25519PublicKey& remote_node_id,
    int64_t capacity,
    int64_t push_amount,
    const ChannelConfig& config) {

    if (capacity < MIN_CHANNEL_CAPACITY) {
        return Result<LightningChannel>::err(
            "Channel capacity below minimum: " + std::to_string(capacity));
    }
    if (capacity > MAX_CHANNEL_CAPACITY) {
        return Result<LightningChannel>::err(
            "Channel capacity exceeds maximum: " + std::to_string(capacity));
    }
    if (push_amount < 0 || push_amount > capacity) {
        return Result<LightningChannel>::err(
            "Invalid push amount: " + std::to_string(push_amount));
    }

    LightningChannel ch;
    ch.state_ = ChannelState::PREOPENING;
    ch.is_funder_ = true;
    ch.temp_channel_id_ = core::get_rand_hash();
    ch.capacity_ = capacity;
    ch.balance_.local = capacity - push_amount;
    ch.balance_.remote = push_amount;
    ch.local_keypair_ = local_keys;
    ch.remote_node_id_ = remote_node_id;

    // Set local config
    ch.local_state_.config = config;
    ch.local_state_.config.channel_reserve = capacity / 100;  // 1% reserve
    ch.local_state_.keys.funding_pubkey = local_keys.public_key;

    LogPrint(LIGHTNING, "Created outbound channel %s, capacity=%lld",
             ch.temp_channel_id_.to_hex().c_str(), capacity);

    return Result<LightningChannel>::ok(std::move(ch));
}

Result<LightningChannel> LightningChannel::create_inbound(
    const crypto::Ed25519KeyPair& local_keys,
    const crypto::Ed25519PublicKey& remote_node_id,
    int64_t capacity,
    int64_t push_amount,
    const ChannelConfig& local_config,
    const ChannelConfig& remote_config) {

    if (capacity < MIN_CHANNEL_CAPACITY) {
        return Result<LightningChannel>::err(
            "Channel capacity below minimum: " + std::to_string(capacity));
    }
    if (capacity > MAX_CHANNEL_CAPACITY) {
        return Result<LightningChannel>::err(
            "Channel capacity exceeds maximum: " + std::to_string(capacity));
    }

    LightningChannel ch;
    ch.state_ = ChannelState::PREOPENING;
    ch.is_funder_ = false;
    ch.temp_channel_id_ = core::get_rand_hash();
    ch.capacity_ = capacity;
    ch.balance_.local = push_amount;
    ch.balance_.remote = capacity - push_amount;
    ch.local_keypair_ = local_keys;
    ch.remote_node_id_ = remote_node_id;

    ch.local_state_.config = local_config;
    ch.local_state_.config.channel_reserve = capacity / 100;
    ch.local_state_.keys.funding_pubkey = local_keys.public_key;
    ch.remote_state_.config = remote_config;

    LogPrint(LIGHTNING, "Created inbound channel %s, capacity=%lld",
             ch.temp_channel_id_.to_hex().c_str(), capacity);

    return Result<LightningChannel>::ok(std::move(ch));
}

// Move constructor/assignment
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

LightningChannel& LightningChannel::operator=(LightningChannel&& other) noexcept {
    if (this != &other) {
        // Lock both in consistent order to prevent deadlock
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

// ── State queries ───────────────────────────────────────────────────

ChannelState LightningChannel::state() const {
    LOCK(mutex_);
    return state_;
}

ChannelId LightningChannel::channel_id() const {
    LOCK(mutex_);
    return channel_id_;
}

TempChannelId LightningChannel::temp_channel_id() const {
    LOCK(mutex_);
    return temp_channel_id_;
}

int64_t LightningChannel::capacity() const {
    LOCK(mutex_);
    return capacity_;
}

ChannelBalance LightningChannel::balance() const {
    LOCK(mutex_);
    return balance_;
}

bool LightningChannel::is_funder() const {
    LOCK(mutex_);
    return is_funder_;
}

const crypto::Ed25519PublicKey& LightningChannel::local_node_id() const {
    return local_keypair_.public_key;
}

const crypto::Ed25519PublicKey& LightningChannel::remote_node_id() const {
    return remote_node_id_;
}

uint64_t LightningChannel::local_commitment_number() const {
    LOCK(mutex_);
    return local_state_.next_commitment_number;
}

uint64_t LightningChannel::remote_commitment_number() const {
    LOCK(mutex_);
    return remote_state_.next_commitment_number;
}

// ── State transitions ───────────────────────────────────────────────

Result<void> LightningChannel::check_state(ChannelState expected) const {
    if (state_ != expected) {
        return Result<void>::err(
            "Invalid channel state: expected " +
            std::string(channel_state_name(expected)) + ", got " +
            std::string(channel_state_name(state_)));
    }
    return Result<void>::ok();
}

Result<void> LightningChannel::check_state_any(
    std::initializer_list<ChannelState> expected) const {
    for (auto s : expected) {
        if (state_ == s) return Result<void>::ok();
    }
    return Result<void>::err(
        "Invalid channel state: " + std::string(channel_state_name(state_)));
}

Result<void> LightningChannel::funding_created(
    const primitives::COutPoint& funding_outpoint,
    const crypto::Ed25519Signature& /*remote_sig*/) {
    LOCK(mutex_);
    auto r = check_state(ChannelState::PREOPENING);
    if (!r) return r;

    funding_outpoint_ = funding_outpoint;
    channel_id_ = make_channel_id(funding_outpoint);
    state_ = ChannelState::FUNDING_CREATED;

    LogPrint(LIGHTNING, "Channel %s: funding created, outpoint=%s",
             channel_id_.to_hex().c_str(),
             funding_outpoint.to_string().c_str());
    return Result<void>::ok();
}

Result<void> LightningChannel::funding_broadcast(const uint256& /*funding_txid*/) {
    LOCK(mutex_);
    auto r = check_state(ChannelState::FUNDING_CREATED);
    if (!r) return r;

    state_ = ChannelState::FUNDING_BROADCAST;

    LogPrint(LIGHTNING, "Channel %s: funding broadcast",
             channel_id_.to_hex().c_str());
    return Result<void>::ok();
}

Result<void> LightningChannel::funding_locked(uint32_t confirmation_height) {
    LOCK(mutex_);
    auto r = check_state(ChannelState::FUNDING_BROADCAST);
    if (!r) return r;

    confirmation_height_ = confirmation_height;
    state_ = ChannelState::FUNDING_LOCKED;

    // Transition immediately to NORMAL after locking
    state_ = ChannelState::NORMAL;

    LogPrint(LIGHTNING, "Channel %s: funding locked at height %u, now NORMAL",
             channel_id_.to_hex().c_str(), confirmation_height);
    return Result<void>::ok();
}

Result<void> LightningChannel::initiate_shutdown() {
    LOCK(mutex_);
    auto r = check_state(ChannelState::NORMAL);
    if (!r) return r;

    state_ = ChannelState::SHUTDOWN;

    LogPrint(LIGHTNING, "Channel %s: shutdown initiated",
             channel_id_.to_hex().c_str());
    return Result<void>::ok();
}

Result<CommitmentTx> LightningChannel::force_close() {
    LOCK(mutex_);
    auto r = check_state_any({ChannelState::NORMAL, ChannelState::SHUTDOWN});
    if (!r) return Result<CommitmentTx>::err(r.error());

    CommitmentTx commit_tx;
    commit_tx.commitment_number = local_state_.next_commitment_number;
    commit_tx.balance = balance_;

    // Build the commitment transaction outputs
    // Output 0: to_local (delayed)
    // Output 1: to_remote (immediate)
    primitives::CMutableTransaction mtx;
    mtx.version = 2;
    mtx.locktime = 0;

    // Input: funding outpoint
    primitives::CTxIn funding_input;
    funding_input.prevout = funding_outpoint_;
    funding_input.sequence = 0xFFFFFFFF;
    mtx.vin.push_back(std::move(funding_input));

    // Output 0: to_local with CSV delay
    if (balance_.local > local_state_.config.dust_limit) {
        primitives::CTxOut local_output;
        local_output.value = balance_.local;
        // Simplified: use P2WPKH for the local output
        local_output.script_pub_key.resize(22);
        local_output.script_pub_key[0] = 0x00;
        local_output.script_pub_key[1] = 0x14;
        std::memcpy(local_output.script_pub_key.data() + 2,
                     local_keypair_.public_key.data.data(), 20);
        mtx.vout.push_back(std::move(local_output));
    }

    // Output 1: to_remote (immediate P2WPKH)
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

    commit_tx.tx = std::move(mtx);
    state_ = ChannelState::FORCE_CLOSING;

    LogPrint(LIGHTNING, "Channel %s: force-closed",
             channel_id_.to_hex().c_str());

    return Result<CommitmentTx>::ok(std::move(commit_tx));
}

Result<void> LightningChannel::mark_closed(CloseType type,
                                            const uint256& closing_txid) {
    LOCK(mutex_);
    close_type_ = type;
    closing_txid_ = closing_txid;
    state_ = ChannelState::CLOSED;

    LogPrint(LIGHTNING, "Channel %s: closed (%s)",
             channel_id_.to_hex().c_str(),
             std::string(close_type_name(type)).c_str());
    return Result<void>::ok();
}

// ── HTLC operations ─────────────────────────────────────────────────

Result<uint64_t> LightningChannel::add_htlc(int64_t amount,
                                             const uint256& payment_hash,
                                             uint32_t cltv_expiry) {
    LOCK(mutex_);
    auto r = check_state(ChannelState::NORMAL);
    if (!r) return Result<uint64_t>::err(r.error());

    Htlc htlc;
    htlc.direction = HtlcDirection::OFFERED;
    htlc.amount = amount;
    htlc.payment_hash = payment_hash;
    htlc.cltv_expiry = cltv_expiry;

    auto vr = validate_htlc(htlc, capacity_, balance_.local,
                             htlcs_.pending_count(),
                             local_state_.config.max_accepted_htlcs,
                             remote_state_.config.min_htlc_value);
    if (!vr) return Result<uint64_t>::err(vr.error());

    uint64_t id = htlcs_.next_id();
    auto ar = htlcs_.add(std::move(htlc));
    if (!ar) return Result<uint64_t>::err(ar.error());

    // Deduct from local balance provisionally
    balance_.local -= amount;

    return Result<uint64_t>::ok(id);
}

Result<void> LightningChannel::fulfill_htlc(uint64_t htlc_id,
                                             const uint256& preimage) {
    LOCK(mutex_);
    auto r = check_state(ChannelState::NORMAL);
    if (!r) return r;

    auto* htlc = htlcs_.find(htlc_id);
    if (!htlc) return Result<void>::err("HTLC not found");
    if (htlc->direction != HtlcDirection::RECEIVED) {
        return Result<void>::err("Can only fulfill received HTLCs");
    }

    int64_t amount = htlc->amount;
    auto fr = htlcs_.fulfill(htlc_id, preimage);
    if (!fr) return fr;

    // Credit to our local balance
    balance_.local += amount;

    return Result<void>::ok();
}

Result<void> LightningChannel::fail_htlc(uint64_t htlc_id) {
    LOCK(mutex_);
    auto r = check_state(ChannelState::NORMAL);
    if (!r) return r;

    auto* htlc = htlcs_.find(htlc_id);
    if (!htlc) return Result<void>::err("HTLC not found");

    int64_t amount = htlc->amount;
    HtlcDirection dir = htlc->direction;

    auto fr = htlcs_.fail(htlc_id);
    if (!fr) return fr;

    // Return funds based on direction
    if (dir == HtlcDirection::OFFERED) {
        balance_.local += amount;
    } else {
        balance_.remote += amount;
    }

    return Result<void>::ok();
}

Result<uint64_t> LightningChannel::receive_htlc(int64_t amount,
                                                  const uint256& payment_hash,
                                                  uint32_t cltv_expiry) {
    LOCK(mutex_);
    auto r = check_state(ChannelState::NORMAL);
    if (!r) return Result<uint64_t>::err(r.error());

    Htlc htlc;
    htlc.direction = HtlcDirection::RECEIVED;
    htlc.amount = amount;
    htlc.payment_hash = payment_hash;
    htlc.cltv_expiry = cltv_expiry;

    auto vr = validate_htlc(htlc, capacity_, balance_.remote,
                             htlcs_.pending_count(),
                             local_state_.config.max_accepted_htlcs,
                             local_state_.config.min_htlc_value);
    if (!vr) return Result<uint64_t>::err(vr.error());

    uint64_t id = htlcs_.next_id();
    auto ar = htlcs_.add(std::move(htlc));
    if (!ar) return Result<uint64_t>::err(ar.error());

    balance_.remote -= amount;

    return Result<uint64_t>::ok(id);
}

// ── Commitment signing ──────────────────────────────────────────────

Result<crypto::Ed25519Signature> LightningChannel::sign_local_commitment() {
    LOCK(mutex_);
    auto r = check_state_any({ChannelState::NORMAL, ChannelState::SHUTDOWN});
    if (!r) return Result<crypto::Ed25519Signature>::err(r.error());

    // Serialize commitment data for signing
    core::DataStream ss;
    core::ser_write_u64(ss, local_state_.next_commitment_number);
    core::Serialize(ss, channel_id_);
    core::ser_write_i64(ss, balance_.local);
    core::ser_write_i64(ss, balance_.remote);

    auto sig_result = crypto::ed25519_sign(
        local_keypair_.secret, ss.span());
    if (!sig_result) {
        return Result<crypto::Ed25519Signature>::err(sig_result.error());
    }

    return Result<crypto::Ed25519Signature>::ok(sig_result.value());
}

Result<void> LightningChannel::receive_commitment_sig(
    const crypto::Ed25519Signature& sig) {
    LOCK(mutex_);
    auto r = check_state_any({ChannelState::NORMAL, ChannelState::SHUTDOWN});
    if (!r) return r;

    // Verify the signature against the remote's public key
    core::DataStream ss;
    core::ser_write_u64(ss, remote_state_.next_commitment_number);
    core::Serialize(ss, channel_id_);
    core::ser_write_i64(ss, balance_.remote);
    core::ser_write_i64(ss, balance_.local);

    if (!crypto::ed25519_verify(remote_node_id_, ss.span(), sig)) {
        return Result<void>::err("Invalid commitment signature from remote");
    }

    remote_state_.next_commitment_number++;
    return Result<void>::ok();
}

Result<uint256> LightningChannel::revoke_and_ack() {
    LOCK(mutex_);
    auto r = check_state_any({ChannelState::NORMAL, ChannelState::SHUTDOWN});
    if (!r) return Result<uint256>::err(r.error());

    // Generate per-commitment secret for the current commitment
    uint256 secret = core::get_rand_hash();
    local_state_.last_per_commitment_secret = secret;
    local_state_.next_commitment_number++;

    return Result<uint256>::ok(secret);
}

Result<void> LightningChannel::receive_revocation(
    const uint256& per_commitment_secret,
    const crypto::Ed25519PublicKey& next_per_commitment_point) {
    LOCK(mutex_);
    auto r = check_state_any({ChannelState::NORMAL, ChannelState::SHUTDOWN});
    if (!r) return r;

    remote_state_.last_per_commitment_secret = per_commitment_secret;
    remote_state_.current_per_commitment_point = next_per_commitment_point;

    // Prune settled HTLCs after successful revocation
    htlcs_.prune_settled();

    return Result<void>::ok();
}

// ── Queries ─────────────────────────────────────────────────────────

const HtlcSet& LightningChannel::htlc_set() const {
    return htlcs_;
}

const ChannelConfig& LightningChannel::local_config() const {
    return local_state_.config;
}

const ChannelConfig& LightningChannel::remote_config() const {
    return remote_state_.config;
}

const primitives::COutPoint& LightningChannel::funding_outpoint() const {
    return funding_outpoint_;
}

uint32_t LightningChannel::pending_htlc_count() const {
    LOCK(mutex_);
    return htlcs_.pending_count();
}

std::string LightningChannel::to_string() const {
    LOCK(mutex_);
    return "Channel{id=" + channel_id_.to_hex() +
           ", state=" + std::string(channel_state_name(state_)) +
           ", capacity=" + std::to_string(capacity_) +
           ", local=" + std::to_string(balance_.local) +
           ", remote=" + std::to_string(balance_.remote) +
           ", htlcs=" + std::to_string(htlcs_.pending_count()) + "}";
}

}  // namespace rnet::lightning
