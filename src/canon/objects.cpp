#include "canon/objects.h"

#include <format>

namespace rnet::canon {

namespace {

// Bounds chosen to reject nonsense early while leaving room for models far larger
// than anything the network can train today.
constexpr uint32_t kMaxDim = 1u << 20;
constexpr uint32_t kMaxLayers = 4096;
constexpr uint32_t kMaxVocab = 1u << 24;
constexpr uint32_t kMaxSeqLen = 1u << 22;

Status ValidateEnum(uint16_t value, uint16_t expected, const char* field) {
    if (value != expected) {
        return Err(std::format("{}: unsupported id {} (this build implements {})", field, value,
                               expected));
    }
    return Status::Ok();
}

}  // namespace

// ---------------------------------------------------------------------------
// ModelSpec
// ---------------------------------------------------------------------------
Status ModelSpec::Validate() const {
    if (d_model == 0 || d_model > kMaxDim) return Err("model: d_model out of range");
    if (n_layers == 0 || n_layers > kMaxLayers) return Err("model: n_layers out of range");
    if (n_heads == 0 || n_heads > d_model) return Err("model: n_heads out of range");
    if (n_kv_heads == 0 || n_kv_heads > n_heads) return Err("model: n_kv_heads out of range");
    if (d_ff == 0 || d_ff > kMaxDim * 8) return Err("model: d_ff out of range");
    if (vocab_size == 0 || vocab_size > kMaxVocab) return Err("model: vocab_size out of range");
    if (seq_len == 0 || seq_len > kMaxSeqLen) return Err("model: seq_len out of range");
    if (rope_theta == 0) return Err("model: rope_theta must be non-zero");
    if (tie_embeddings > 1) return Err("model: tie_embeddings must be 0 or 1");
    if (qk_norm > 1) return Err("model: qk_norm must be 0 or 1");
    if (d_model % n_heads != 0) return Err("model: d_model not divisible by n_heads");
    if (n_heads % n_kv_heads != 0) return Err("model: n_heads not divisible by n_kv_heads (GQA)");
    if (auto st = ValidateEnum(static_cast<uint16_t>(norm),
                               static_cast<uint16_t>(NormId::RmsNorm), "model.norm");
        !st) {
        return st;
    }
    return ValidateEnum(static_cast<uint16_t>(activation),
                        static_cast<uint16_t>(ActivationId::SwiGlu), "model.activation");
}

uint64_t ModelSpec::ParameterCount() const {
    const uint64_t d = d_model;
    const uint64_t head_dim = d / n_heads;
    const uint64_t attn = d * (n_heads * head_dim)          // wq
                          + d * (n_kv_heads * head_dim)     // wk
                          + d * (n_kv_heads * head_dim)     // wv
                          + (n_heads * head_dim) * d;       // wo
    const uint64_t ffn = 3ull * d * d_ff;                   // gate, up, down
    const uint64_t norms = 2ull * d + (qk_norm ? 2ull * head_dim : 0ull);
    const uint64_t per_layer = attn + ffn + norms;
    const uint64_t embed = static_cast<uint64_t>(vocab_size) * d;
    const uint64_t head = tie_embeddings ? 0ull : embed;
    return per_layer * n_layers + embed + head + d;         // + final norm
}

// ---------------------------------------------------------------------------
// RoundDescriptor
// ---------------------------------------------------------------------------
std::vector<uint8_t> RoundDescriptor::Serialize() const {
    Writer w;
    w.U32(protocol_version);
    w.U32(network_magic);
    w.U64(round_id);
    w.U32(model.d_model);
    w.U32(model.n_layers);
    w.U32(model.n_heads);
    w.U32(model.n_kv_heads);
    w.U32(model.d_ff);
    w.U32(model.vocab_size);
    w.U32(model.seq_len);
    w.U32(model.rope_theta);
    w.U8(model.tie_embeddings);
    w.U8(model.qk_norm);
    w.U16(static_cast<uint16_t>(model.norm));
    w.U16(static_cast<uint16_t>(model.activation));
    w.U16(static_cast<uint16_t>(determinism_class));
    w.U16(static_cast<uint16_t>(optimizer));
    w.Hash(tokenizer_hash);
    w.Hash(dataset_root);
    return w.take();
}

Result<RoundDescriptor> RoundDescriptor::Deserialize(std::span<const uint8_t> content) {
    Reader r(content);
    RoundDescriptor d;

#define RNET_READ(field, call)                       \
    {                                                \
        auto v = call;                               \
        if (!v) return Err(v.error());               \
        field = v.value();                           \
    }

    RNET_READ(d.protocol_version, r.U32())
    RNET_READ(d.network_magic, r.U32())
    RNET_READ(d.round_id, r.U64())
    RNET_READ(d.model.d_model, r.U32())
    RNET_READ(d.model.n_layers, r.U32())
    RNET_READ(d.model.n_heads, r.U32())
    RNET_READ(d.model.n_kv_heads, r.U32())
    RNET_READ(d.model.d_ff, r.U32())
    RNET_READ(d.model.vocab_size, r.U32())
    RNET_READ(d.model.seq_len, r.U32())
    RNET_READ(d.model.rope_theta, r.U32())
    RNET_READ(d.model.tie_embeddings, r.U8())
    RNET_READ(d.model.qk_norm, r.U8())

    uint16_t norm = 0, act = 0, det = 0, opt = 0;
    RNET_READ(norm, r.U16())
    RNET_READ(act, r.U16())
    RNET_READ(det, r.U16())
    RNET_READ(opt, r.U16())
    RNET_READ(d.tokenizer_hash, r.Hash())
    RNET_READ(d.dataset_root, r.Hash())
#undef RNET_READ

    d.model.norm = static_cast<NormId>(norm);
    d.model.activation = static_cast<ActivationId>(act);
    d.determinism_class = static_cast<DeterminismClass>(det);
    d.optimizer = static_cast<OptimizerId>(opt);

    if (auto st = r.ExpectExhausted(); !st) return Err(st.error());
    if (d.protocol_version != kProtocolVersion) {
        return Err("round: unsupported protocol version");
    }
    if (auto st = d.model.Validate(); !st) return Err(st.error());
    if (auto st = ValidateEnum(opt, static_cast<uint16_t>(OptimizerId::Adafactor), "round.optimizer");
        !st) {
        return Err(st.error());
    }
    return d;
}

std::vector<uint8_t> RoundDescriptor::ToContainer() const {
    return WrapContainer(ObjectType::RoundDescriptor, Serialize());
}

Result<RoundDescriptor> RoundDescriptor::FromContainer(std::span<const uint8_t> raw) {
    auto c = ParseContainer(raw);
    if (!c) return Err(c.error());
    if (c.value().type != ObjectType::RoundDescriptor) {
        return Err("round: container holds " + ObjectTypeName(c.value().type));
    }
    return Deserialize(c.value().content);
}

crypto::Hash256 RoundDescriptor::Id() const { return ContainerId(ToContainer()); }

// ---------------------------------------------------------------------------
// DatasetManifest
// ---------------------------------------------------------------------------
Status DatasetManifest::Validate() const {
    if (n_tokens == 0) return Err("dataset: empty corpus");
    if (chunk_tokens == 0) return Err("dataset: chunk_tokens must be non-zero");
    if (n_chunks == 0) return Err("dataset: n_chunks must be non-zero");
    if (!IsKnownTokenDtype(dtype)) return Err("dataset: unknown token dtype");
    // n_chunks must be exactly ceil(n_tokens / chunk_tokens): anything else means
    // the manifest and the file disagree about the tree shape, and therefore about
    // the root.
    const uint64_t expected = (n_tokens + chunk_tokens - 1) / chunk_tokens;
    if (expected != n_chunks) {
        return Err(std::format("dataset: n_chunks {} != ceil(n_tokens/chunk_tokens) {}", n_chunks,
                               expected));
    }
    return Status::Ok();
}

std::vector<uint8_t> DatasetManifest::Serialize() const {
    Writer w;
    w.Hash(dataset_root);
    w.Hash(tokenizer_hash);
    w.U64(n_tokens);
    w.U32(chunk_tokens);
    w.U32(n_chunks);
    w.U8(static_cast<uint8_t>(dtype));
    return w.take();
}

Result<DatasetManifest> DatasetManifest::Deserialize(std::span<const uint8_t> content) {
    Reader r(content);
    DatasetManifest m;

    auto root = r.Hash();
    if (!root) return Err(root.error());
    m.dataset_root = root.value();

    auto tok = r.Hash();
    if (!tok) return Err(tok.error());
    m.tokenizer_hash = tok.value();

    auto n_tokens = r.U64();
    if (!n_tokens) return Err(n_tokens.error());
    m.n_tokens = n_tokens.value();

    auto chunk = r.U32();
    if (!chunk) return Err(chunk.error());
    m.chunk_tokens = chunk.value();

    auto n_chunks = r.U32();
    if (!n_chunks) return Err(n_chunks.error());
    m.n_chunks = n_chunks.value();

    auto dtype = r.U8();
    if (!dtype) return Err(dtype.error());
    m.dtype = static_cast<TokenDtype>(dtype.value());

    if (auto st = r.ExpectExhausted(); !st) return Err(st.error());
    if (auto st = m.Validate(); !st) return Err(st.error());
    return m;
}

std::vector<uint8_t> DatasetManifest::ToContainer() const {
    return WrapContainer(ObjectType::DatasetManifest, Serialize());
}

Result<DatasetManifest> DatasetManifest::FromContainer(std::span<const uint8_t> raw) {
    auto c = ParseContainer(raw);
    if (!c) return Err(c.error());
    if (c.value().type != ObjectType::DatasetManifest) {
        return Err("dataset: container holds " + ObjectTypeName(c.value().type));
    }
    return Deserialize(c.value().content);
}

crypto::Hash256 DatasetManifest::Id() const { return ContainerId(ToContainer()); }

// ---------------------------------------------------------------------------
// ContributionHeader
// ---------------------------------------------------------------------------
Status ContributionHeader::Validate() const {
    if (n_params == 0) return Err("contribution: n_params must be non-zero");
    if (inner_steps == 0) return Err("contribution: inner_steps must be non-zero");
    // The exponent bound keeps 2^-scale_exp representable and stops a hostile
    // header from forcing an absurd shift during dequantisation.
    if (scale_exp < -64 || scale_exp > 64) return Err("contribution: scale_exp out of range");
    if (base_checkpoint == crypto::Hash256{}) {
        return Err("contribution: base_checkpoint must name the weights it started from");
    }
    return Status::Ok();
}

std::vector<uint8_t> ContributionHeader::Serialize() const {
    Writer w;
    w.U64(round_id);
    w.U64(worker_id);
    w.U64(outer_step);
    w.Hash(base_checkpoint);
    w.Hash(payload_hash);
    w.U64(n_params);
    w.U32(inner_steps);
    w.U32(static_cast<uint32_t>(scale_exp));   // two's complement, fixed width
    return w.take();
}

Result<ContributionHeader> ContributionHeader::Deserialize(std::span<const uint8_t> content) {
    Reader r(content);
    ContributionHeader h;

    auto round_id = r.U64();
    if (!round_id) return Err(round_id.error());
    h.round_id = round_id.value();

    auto worker_id = r.U64();
    if (!worker_id) return Err(worker_id.error());
    h.worker_id = worker_id.value();

    auto outer_step = r.U64();
    if (!outer_step) return Err(outer_step.error());
    h.outer_step = outer_step.value();

    auto base = r.Hash();
    if (!base) return Err(base.error());
    h.base_checkpoint = base.value();

    auto payload = r.Hash();
    if (!payload) return Err(payload.error());
    h.payload_hash = payload.value();

    auto n_params = r.U64();
    if (!n_params) return Err(n_params.error());
    h.n_params = n_params.value();

    auto inner = r.U32();
    if (!inner) return Err(inner.error());
    h.inner_steps = inner.value();

    auto scale = r.U32();
    if (!scale) return Err(scale.error());
    h.scale_exp = static_cast<int32_t>(scale.value());

    if (auto st = r.ExpectExhausted(); !st) return Err(st.error());
    if (auto st = h.Validate(); !st) return Err(st.error());
    return h;
}

std::vector<uint8_t> ContributionHeader::ToContainer() const {
    return WrapContainer(ObjectType::Contribution, Serialize());
}

Result<ContributionHeader> ContributionHeader::FromContainer(std::span<const uint8_t> raw) {
    auto c = ParseContainer(raw);
    if (!c) return Err(c.error());
    if (c.value().type != ObjectType::Contribution) {
        return Err("contribution: container holds " + ObjectTypeName(c.value().type));
    }
    return Deserialize(c.value().content);
}

crypto::Hash256 ContributionHeader::Id() const { return ContainerId(ToContainer()); }

// ---------------------------------------------------------------------------
// CheckpointHeader
// ---------------------------------------------------------------------------
Status CheckpointHeader::Validate() const {
    if (n_params == 0) return Err("checkpoint: n_params must be non-zero");
    if (weight_dtype != 1 && weight_dtype != 2) return Err("checkpoint: unknown weight dtype");
    if (weights_hash == crypto::Hash256{}) return Err("checkpoint: weights_hash must be set");
    // Only the genesis checkpoint may have no parent, and it must be step zero.
    if (IsGenesis() && outer_step != 0) {
        return Err("checkpoint: parentless checkpoint must be outer_step 0");
    }
    if (!IsGenesis() && outer_step == 0) {
        return Err("checkpoint: outer_step 0 must be the parentless genesis checkpoint");
    }
    if (!IsGenesis() && contributor_count == 0) {
        return Err("checkpoint: a non-genesis checkpoint must aggregate at least one contribution");
    }
    // A checkpoint that aggregated something must say what. Otherwise it states an
    // outcome nobody can recompute, and a fabricated checkpoint is indistinguishable
    // from an honest one.
    if (!IsGenesis() && evidence_hash == crypto::Hash256{}) {
        return Err("checkpoint: a non-genesis checkpoint must name the contributions it aggregated");
    }
    if (IsGenesis() && evidence_hash != crypto::Hash256{}) {
        return Err("checkpoint: the genesis checkpoint aggregates nothing and must carry no evidence");
    }
    // Genesis precedes the optimizer's existence; every later checkpoint is the
    // result of a step, and a step always leaves a state. An all-zero value here
    // means someone forgot to record it, and accepting that would put the
    // producer's momentum beyond anyone's ability to check.
    if (IsGenesis() && optimizer_state_hash != crypto::Hash256{}) {
        return Err("checkpoint: the genesis checkpoint has no optimizer state to record");
    }
    if (!IsGenesis() && optimizer_state_hash == crypto::Hash256{}) {
        return Err("checkpoint: a non-genesis checkpoint must record the optimizer state it "
                   "produced, or the update it published cannot be verified by anyone");
    }
    return Status::Ok();
}

std::vector<uint8_t> CheckpointHeader::Serialize() const {
    Writer w;
    w.Hash(parent);
    w.U64(round_id);
    w.U64(outer_step);
    w.Hash(weights_hash);
    w.U64(n_params);
    w.U32(contributor_count);
    w.Hash(evidence_hash);
    w.Hash(optimizer_state_hash);
    w.U8(weight_dtype);
    return w.take();
}

Result<CheckpointHeader> CheckpointHeader::Deserialize(std::span<const uint8_t> content) {
    Reader r(content);
    CheckpointHeader h;

    auto parent = r.Hash();
    if (!parent) return Err(parent.error());
    h.parent = parent.value();

    auto round_id = r.U64();
    if (!round_id) return Err(round_id.error());
    h.round_id = round_id.value();

    auto outer_step = r.U64();
    if (!outer_step) return Err(outer_step.error());
    h.outer_step = outer_step.value();

    auto weights = r.Hash();
    if (!weights) return Err(weights.error());
    h.weights_hash = weights.value();

    auto n_params = r.U64();
    if (!n_params) return Err(n_params.error());
    h.n_params = n_params.value();

    auto contributors = r.U32();
    if (!contributors) return Err(contributors.error());
    h.contributor_count = contributors.value();

    auto evidence = r.Hash();
    if (!evidence) return Err(evidence.error());
    h.evidence_hash = evidence.value();

    auto optimizer_state = r.Hash();
    if (!optimizer_state) return Err(optimizer_state.error());
    h.optimizer_state_hash = optimizer_state.value();

    auto dtype = r.U8();
    if (!dtype) return Err(dtype.error());
    h.weight_dtype = dtype.value();

    if (auto st = r.ExpectExhausted(); !st) return Err(st.error());
    if (auto st = h.Validate(); !st) return Err(st.error());
    return h;
}

std::vector<uint8_t> CheckpointHeader::ToContainer() const {
    return WrapContainer(ObjectType::Checkpoint, Serialize());
}

Result<CheckpointHeader> CheckpointHeader::FromContainer(std::span<const uint8_t> raw) {
    auto c = ParseContainer(raw);
    if (!c) return Err(c.error());
    if (c.value().type != ObjectType::Checkpoint) {
        return Err("checkpoint: container holds " + ObjectTypeName(c.value().type));
    }
    return Deserialize(c.value().content);
}

crypto::Hash256 CheckpointHeader::Id() const { return ContainerId(ToContainer()); }

// ---------------------------------------------------------------------------
// PolicyDescriptor
// ---------------------------------------------------------------------------
Status PolicyDescriptor::Validate() const {
    if (protocol_version != kProtocolVersion) return Err("policy: unsupported protocol version");
    if (inner_steps == 0) return Err("policy: inner_steps must be non-zero");
    if (micro_batch == 0) return Err("policy: micro_batch must be non-zero");
    if (min_contributors == 0) return Err("policy: min_contributors must be non-zero");
    if (checkpoint_interval == 0) return Err("policy: checkpoint_interval must be non-zero");
    if (dataset_chunk_tokens == 0) return Err("policy: dataset_chunk_tokens must be non-zero");
    if (outer_momentum_q16 >= (1u << 16)) return Err("policy: outer momentum must be in [0, 1)");
    if (outer_lr_q16 == 0 || outer_lr_q16 > (4u << 16)) {
        return Err("policy: outer learning rate must be in (0, 4]");
    }
    if (nesterov > 1) return Err("policy: nesterov must be 0 or 1");
    if (shadow_mode > 1) return Err("policy: shadow_mode must be 0 or 1");
    if (challenge_percent == 0 || challenge_percent > 100) {
        return Err("policy: challenge_percent must be in 1..100");
    }
    if (retained_checkpoints == 0) return Err("policy: retained_checkpoints must be non-zero");
    if (challenge_deadline_steps == 0) return Err("policy: challenge_deadline_steps must be non-zero");
    // The load-bearing invariant of verification: a challenge must expire before
    // the weights needed to answer it are pruned, or it is unanswerable by
    // construction and an accuser could force a verdict nobody can rebut.
    if (challenge_deadline_steps >= retained_checkpoints) {
        return Err(std::format(
            "policy: challenge_deadline_steps ({}) must be < retained_checkpoints ({}) — a "
            "challenge whose base weights have been pruned cannot be answered",
            challenge_deadline_steps, retained_checkpoints));
    }
    if (slash_quorum == 0) return Err("policy: slash_quorum must be non-zero");
    return Status::Ok();
}

std::vector<uint8_t> PolicyDescriptor::Serialize() const {
    Writer w;
    w.U32(protocol_version);
    w.U32(network_magic);
    w.U64(policy_version);
    w.U32(inner_steps);
    w.U32(micro_batch);
    w.U32(min_contributors);
    w.U32(checkpoint_interval);
    w.U32(dataset_chunk_tokens);
    w.U32(round_deadline_ms);
    w.U32(outer_momentum_q16);
    w.U32(outer_lr_q16);
    w.U8(nesterov);
    w.U32(challenge_percent);
    w.U32(challenge_deadline_steps);
    w.U32(retained_checkpoints);
    w.U32(slash_quorum);
    w.U8(shadow_mode);
    return w.take();
}

Result<PolicyDescriptor> PolicyDescriptor::Deserialize(std::span<const uint8_t> content) {
    Reader r(content);
    PolicyDescriptor p;

#define RNET_READ(field, call)         \
    {                                  \
        auto v = call;                 \
        if (!v) return Err(v.error()); \
        field = v.value();             \
    }
    RNET_READ(p.protocol_version, r.U32())
    RNET_READ(p.network_magic, r.U32())
    RNET_READ(p.policy_version, r.U64())
    RNET_READ(p.inner_steps, r.U32())
    RNET_READ(p.micro_batch, r.U32())
    RNET_READ(p.min_contributors, r.U32())
    RNET_READ(p.checkpoint_interval, r.U32())
    RNET_READ(p.dataset_chunk_tokens, r.U32())
    RNET_READ(p.round_deadline_ms, r.U32())
    RNET_READ(p.outer_momentum_q16, r.U32())
    RNET_READ(p.outer_lr_q16, r.U32())
    RNET_READ(p.nesterov, r.U8())
    RNET_READ(p.challenge_percent, r.U32())
    RNET_READ(p.challenge_deadline_steps, r.U32())
    RNET_READ(p.retained_checkpoints, r.U32())
    RNET_READ(p.slash_quorum, r.U32())
    RNET_READ(p.shadow_mode, r.U8())
#undef RNET_READ

    if (auto st = r.ExpectExhausted(); !st) return Err(st.error());
    if (auto st = p.Validate(); !st) return Err(st.error());
    return p;
}

std::vector<uint8_t> PolicyDescriptor::ToContainer() const {
    return WrapContainer(ObjectType::PolicyDescriptor, Serialize());
}

Result<PolicyDescriptor> PolicyDescriptor::FromContainer(std::span<const uint8_t> raw) {
    auto c = ParseContainer(raw);
    if (!c) return Err(c.error());
    if (c.value().type != ObjectType::PolicyDescriptor) {
        return Err("policy: container holds " + ObjectTypeName(c.value().type));
    }
    return Deserialize(c.value().content);
}

crypto::Hash256 PolicyDescriptor::Id() const { return ContainerId(ToContainer()); }

// ---------------------------------------------------------------------------
// ChallengeOrder
// ---------------------------------------------------------------------------
Status ChallengeOrder::Validate() const {
    if (contribution_id == crypto::Hash256{}) return Err("challenge: contribution_id must be set");
    // A zero beacon would mean the selection was not driven by external
    // randomness, which is exactly the grinding hole this field exists to close.
    if (beacon == crypto::Hash256{}) return Err("challenge: beacon must be set");
    if (issued_at_step < outer_step) {
        return Err("challenge: cannot be issued before the step it challenges");
    }
    return Status::Ok();
}

std::vector<uint8_t> ChallengeOrder::Serialize() const {
    Writer w;
    w.U64(round_id);
    w.U64(outer_step);
    w.Hash(contribution_id);
    w.Hash(beacon);
    w.U64(beacon_height);
    w.U64(verifier_id);
    w.U64(issued_at_step);
    w.U16(determinism_class);
    return w.take();
}

Result<ChallengeOrder> ChallengeOrder::Deserialize(std::span<const uint8_t> content) {
    Reader r(content);
    ChallengeOrder c;

#define RNET_READ(field, call)         \
    {                                  \
        auto v = call;                 \
        if (!v) return Err(v.error()); \
        field = v.value();             \
    }
    RNET_READ(c.round_id, r.U64())
    RNET_READ(c.outer_step, r.U64())
    RNET_READ(c.contribution_id, r.Hash())
    RNET_READ(c.beacon, r.Hash())
    RNET_READ(c.beacon_height, r.U64())
    RNET_READ(c.verifier_id, r.U64())
    RNET_READ(c.issued_at_step, r.U64())
    RNET_READ(c.determinism_class, r.U16())
#undef RNET_READ

    if (auto st = r.ExpectExhausted(); !st) return Err(st.error());
    if (auto st = c.Validate(); !st) return Err(st.error());
    return c;
}

std::vector<uint8_t> ChallengeOrder::ToContainer() const {
    return WrapContainer(ObjectType::Challenge, Serialize());
}

Result<ChallengeOrder> ChallengeOrder::FromContainer(std::span<const uint8_t> raw) {
    auto c = ParseContainer(raw);
    if (!c) return Err(c.error());
    if (c.value().type != ObjectType::Challenge) {
        return Err("challenge: container holds " + ObjectTypeName(c.value().type));
    }
    return Deserialize(c.value().content);
}

crypto::Hash256 ChallengeOrder::Id() const { return ContainerId(ToContainer()); }

// ---------------------------------------------------------------------------
// VerificationVerdict
// ---------------------------------------------------------------------------
Status VerificationVerdict::Validate() const {
    if (challenge_id == crypto::Hash256{}) return Err("verdict: challenge_id must be set");
    if (contribution_id == crypto::Hash256{}) return Err("verdict: contribution_id must be set");
    switch (kind) {
        case VerdictKind::Match:
            // A match must show that the two hashes really are equal; otherwise a
            // verifier could vouch for work it never reproduced.
            if (recomputed_payload_hash != claimed_payload_hash) {
                return Err("verdict: Match with differing payload hashes");
            }
            return Status::Ok();
        case VerdictKind::Mismatch:
            if (recomputed_payload_hash == claimed_payload_hash) {
                return Err("verdict: Mismatch with identical payload hashes");
            }
            return Status::Ok();
        case VerdictKind::Unavailable:
        case VerdictKind::OutOfClass:
            return Status::Ok();
    }
    return Err("verdict: unknown kind");
}

std::vector<uint8_t> VerificationVerdict::Serialize() const {
    Writer w;
    w.Hash(challenge_id);
    w.Hash(contribution_id);
    w.U64(verifier_id);
    w.U16(static_cast<uint16_t>(kind));
    w.Hash(claimed_payload_hash);
    w.Hash(recomputed_payload_hash);
    w.U16(determinism_class);
    w.U64(reported_at_step);
    return w.take();
}

Result<VerificationVerdict> VerificationVerdict::Deserialize(std::span<const uint8_t> content) {
    Reader r(content);
    VerificationVerdict v;

    auto challenge = r.Hash();
    if (!challenge) return Err(challenge.error());
    v.challenge_id = challenge.value();

    auto contribution = r.Hash();
    if (!contribution) return Err(contribution.error());
    v.contribution_id = contribution.value();

    auto verifier = r.U64();
    if (!verifier) return Err(verifier.error());
    v.verifier_id = verifier.value();

    auto kind = r.U16();
    if (!kind) return Err(kind.error());
    if (kind.value() < 1 || kind.value() > 4) return Err("verdict: unknown kind");
    v.kind = static_cast<VerdictKind>(kind.value());

    auto claimed = r.Hash();
    if (!claimed) return Err(claimed.error());
    v.claimed_payload_hash = claimed.value();

    auto recomputed = r.Hash();
    if (!recomputed) return Err(recomputed.error());
    v.recomputed_payload_hash = recomputed.value();

    auto det = r.U16();
    if (!det) return Err(det.error());
    v.determinism_class = det.value();

    auto step = r.U64();
    if (!step) return Err(step.error());
    v.reported_at_step = step.value();

    if (auto st = r.ExpectExhausted(); !st) return Err(st.error());
    if (auto st = v.Validate(); !st) return Err(st.error());
    return v;
}

std::vector<uint8_t> VerificationVerdict::ToContainer() const {
    return WrapContainer(ObjectType::Verdict, Serialize());
}

Result<VerificationVerdict> VerificationVerdict::FromContainer(std::span<const uint8_t> raw) {
    auto c = ParseContainer(raw);
    if (!c) return Err(c.error());
    if (c.value().type != ObjectType::Verdict) {
        return Err("verdict: container holds " + ObjectTypeName(c.value().type));
    }
    return Deserialize(c.value().content);
}

crypto::Hash256 VerificationVerdict::Id() const { return ContainerId(ToContainer()); }

// ---------------------------------------------------------------------------
// SlashEvidence
// ---------------------------------------------------------------------------
Status SlashEvidence::Validate() const {
    if (contribution_id == crypto::Hash256{}) return Err("slash: contribution_id must be set");
    if (verdict_ids.empty()) return Err("slash: evidence requires at least one verdict");
    if (verdict_ids.size() > 1024) return Err("slash: implausible number of verdicts");
    // Sorted and unique: the same verdict counted twice would let one verifier
    // manufacture a quorum on its own.
    for (size_t i = 1; i < verdict_ids.size(); ++i) {
        if (!(verdict_ids[i - 1] < verdict_ids[i])) {
            return Err("slash: verdict ids must be sorted and unique");
        }
    }
    return Status::Ok();
}

std::vector<uint8_t> SlashEvidence::Serialize() const {
    Writer w;
    w.Hash(contribution_id);
    w.U64(worker_id);
    w.U64(round_id);
    w.U32(static_cast<uint32_t>(verdict_ids.size()));
    for (const auto& id : verdict_ids) w.Hash(id);
    w.U16(determinism_class);
    w.U64(finalized_at_step);
    return w.take();
}

Result<SlashEvidence> SlashEvidence::Deserialize(std::span<const uint8_t> content) {
    Reader r(content);
    SlashEvidence e;

    auto contribution = r.Hash();
    if (!contribution) return Err(contribution.error());
    e.contribution_id = contribution.value();

    auto worker = r.U64();
    if (!worker) return Err(worker.error());
    e.worker_id = worker.value();

    auto round_id = r.U64();
    if (!round_id) return Err(round_id.error());
    e.round_id = round_id.value();

    auto count = r.U32();
    if (!count) return Err(count.error());
    if (count.value() > 1024) return Err("slash: implausible number of verdicts");
    for (uint32_t i = 0; i < count.value(); ++i) {
        auto id = r.Hash();
        if (!id) return Err(id.error());
        e.verdict_ids.push_back(id.value());
    }

    auto det = r.U16();
    if (!det) return Err(det.error());
    e.determinism_class = det.value();

    auto step = r.U64();
    if (!step) return Err(step.error());
    e.finalized_at_step = step.value();

    if (auto st = r.ExpectExhausted(); !st) return Err(st.error());
    if (auto st = e.Validate(); !st) return Err(st.error());
    return e;
}

std::vector<uint8_t> SlashEvidence::ToContainer() const {
    return WrapContainer(ObjectType::SlashEvidence, Serialize());
}

Result<SlashEvidence> SlashEvidence::FromContainer(std::span<const uint8_t> raw) {
    auto c = ParseContainer(raw);
    if (!c) return Err(c.error());
    if (c.value().type != ObjectType::SlashEvidence) {
        return Err("slash: container holds " + ObjectTypeName(c.value().type));
    }
    return Deserialize(c.value().content);
}

crypto::Hash256 SlashEvidence::Id() const { return ContainerId(ToContainer()); }

// ---------------------------------------------------------------------------
// BatchSeed
// ---------------------------------------------------------------------------
std::vector<uint8_t> BatchSeed::Serialize() const {
    Writer w;
    w.Hash(dataset_root);
    w.U64(round_id);
    w.U64(worker_id);
    w.U64(step);
    w.U32(index);
    return w.take();
}

std::vector<uint8_t> BatchSeed::ToContainer() const {
    return WrapContainer(ObjectType::BatchSeed, Serialize());
}

crypto::Hash256 BatchSeed::Id() const { return ContainerId(ToContainer()); }

}  // namespace rnet::canon
