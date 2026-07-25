// The typed consensus objects.
//
// A field belongs here if two honest nodes must agree on it. Anything that
// affects the computation (every model dimension, the tokenizer, the optimizer,
// the corpus) is part of the object, because if nodes disagree their gradients
// and checkpoints are incompatible — the training equivalent of a chain split.
#pragma once

#include <cstdint>
#include <string>

#include "canon/canon.h"
#include "crypto/sha3.h"
#include "util/result.h"

namespace rnet::canon {

// Registries. Values are permanent: once a round is published with norm_id=1,
// that number means RMSNorm forever.
enum class NormId : uint16_t { RmsNorm = 1 };
enum class ActivationId : uint16_t { SwiGlu = 1 };
enum class OptimizerId : uint16_t { Adafactor = 1 };
enum class TokenDtype : uint8_t { Uint16 = 1, Uint32 = 2 };

// Determinism class: which hardware/kernel configuration a worker's arithmetic
// belongs to. Spot-recompute verification only holds between workers of the same
// class. 0 = unassigned, pending the cross-architecture determinism measurement.
enum class DeterminismClass : uint16_t { Pending = 0 };

// A complete model architecture. No dimension may live anywhere else in the code.
struct ModelSpec {
    uint32_t d_model{};
    uint32_t n_layers{};
    uint32_t n_heads{};
    uint32_t n_kv_heads{};
    uint32_t d_ff{};
    uint32_t vocab_size{};
    uint32_t seq_len{};
    uint32_t rope_theta{};       // integer base: exact, no float bit patterns on the wire
    uint8_t tie_embeddings{};
    uint8_t qk_norm{};
    NormId norm{NormId::RmsNorm};
    ActivationId activation{ActivationId::SwiGlu};

    // Structural validity (divisibility, non-zero, sane bounds). A descriptor that
    // fails this can never be trained consistently and is rejected on load.
    Status Validate() const;

    // Parameter count for the described model, used for sizing and for sanity
    // checks against checkpoints.
    uint64_t ParameterCount() const;
};

// One training round: what to train, how, and on what data.
struct RoundDescriptor {
    uint32_t protocol_version{kProtocolVersion};
    // Binds the descriptor to one network. Without it, two networks that happen to
    // train the same architecture would produce byte-identical genesis artifacts,
    // and an artifact from one would be accepted by the other.
    uint32_t network_magic{};
    uint64_t round_id{};
    ModelSpec model{};
    DeterminismClass determinism_class{DeterminismClass::Pending};
    OptimizerId optimizer{OptimizerId::Adafactor};
    crypto::Hash256 tokenizer_hash{};   // all-zero = byte-level, no artifact
    crypto::Hash256 dataset_root{};     // all-zero = no corpus pinned yet

    std::vector<uint8_t> Serialize() const;
    static Result<RoundDescriptor> Deserialize(std::span<const uint8_t> content);

    std::vector<uint8_t> ToContainer() const;
    static Result<RoundDescriptor> FromContainer(std::span<const uint8_t> raw);
    crypto::Hash256 Id() const;         // SHA3-256 of the container = round hash
};

// Describes a tokenized corpus: its Merkle root, how it is chunked, and which
// tokenizer produced it.
struct DatasetManifest {
    crypto::Hash256 dataset_root{};
    crypto::Hash256 tokenizer_hash{};
    uint64_t n_tokens{};
    uint32_t chunk_tokens{};            // tokens per Merkle leaf
    uint32_t n_chunks{};
    TokenDtype dtype{TokenDtype::Uint32};

    Status Validate() const;
    std::vector<uint8_t> Serialize() const;
    static Result<DatasetManifest> Deserialize(std::span<const uint8_t> content);

    std::vector<uint8_t> ToContainer() const;
    static Result<DatasetManifest> FromContainer(std::span<const uint8_t> raw);
    crypto::Hash256 Id() const;
};

// One worker's contribution to an outer step: the header that is hashed, signed
// and gossiped. The quantised payload itself travels separately (it is large);
// `payload_hash` binds the two, so a header can be validated before deciding to
// download megabytes.
struct ContributionHeader {
    uint64_t round_id{};
    uint64_t worker_id{};
    uint64_t outer_step{};
    crypto::Hash256 base_checkpoint{};   // weights this contribution was computed from
    crypto::Hash256 payload_hash{};      // SHA3-256 of the int8 payload
    uint64_t n_params{};
    uint32_t inner_steps{};              // local steps actually performed
    int32_t scale_exp{};                 // payload units are 2^-scale_exp (see diloco/quantize)

    Status Validate() const;
    std::vector<uint8_t> Serialize() const;
    static Result<ContributionHeader> Deserialize(std::span<const uint8_t> content);

    std::vector<uint8_t> ToContainer() const;
    static Result<ContributionHeader> FromContainer(std::span<const uint8_t> raw);
    crypto::Hash256 Id() const;
};

// A published model. Checkpoints form a chain: each names its parent, so any node
// can prove a model descends from genesis by following 32-byte links, without
// holding the intermediate weights.
struct CheckpointHeader {
    crypto::Hash256 parent{};            // all-zero only for the genesis checkpoint
    uint64_t round_id{};
    uint64_t outer_step{};
    crypto::Hash256 weights_hash{};      // SHA3-256 of the canonical weight bytes
    uint64_t n_params{};
    uint32_t contributor_count{};        // how many contributions were aggregated

    // WHICH contributions were aggregated: SHA3-256 over their ids in ascending
    // worker order. Without it a checkpoint states an outcome and names no inputs,
    // so nobody can recompute the aggregation and nobody can tell a correct
    // checkpoint from a fabricated one. It is what makes the producer's work
    // checkable rather than merely announced.
    crypto::Hash256 evidence_hash{};

    uint8_t weight_dtype{};              // 1 = bf16, 2 = fp32

    bool IsGenesis() const { return parent == crypto::Hash256{}; }

    Status Validate() const;
    std::vector<uint8_t> Serialize() const;
    static Result<CheckpointHeader> Deserialize(std::span<const uint8_t> content);

    std::vector<uint8_t> ToContainer() const;
    static Result<CheckpointHeader> FromContainer(std::span<const uint8_t> raw);
    crypto::Hash256 Id() const;
};

// Operational policy: how the network schedules, verifies and settles.
//
// Deliberately separate from RoundDescriptor because the two change for different
// reasons. Model dimensions are frozen until a hard fork; a challenge rate or a
// quorum may need tuning once real network data arrives. Folding them together
// would mean re-emitting — and re-pinning — the model's genesis every time an
// operational knob moved.
//
// Two artifacts, two reasons to update, one trust anchor each.
struct PolicyDescriptor {
    uint32_t protocol_version{kProtocolVersion};
    uint32_t network_magic{};      // binds the policy to one network, as the round is
    uint64_t policy_version{};     // monotonic: lets a network adopt a new policy in place

    // Training schedule.
    uint32_t inner_steps{};
    uint32_t micro_batch{};
    uint32_t min_contributors{};
    uint32_t checkpoint_interval{};
    uint32_t dataset_chunk_tokens{};

    // How long a round stays open, in milliseconds. A consensus value rather than
    // a local setting: it decides how long a worker has to arrive before the round
    // closes without it, so two nodes using different numbers would disagree about
    // who was late — and produce different checkpoints from the same network.
    uint32_t round_deadline_ms{};

    // Outer optimizer, in Q16 fixed point so every node steps identically.
    uint32_t outer_momentum_q16{};
    uint32_t outer_lr_q16{};
    uint8_t nesterov{};

    // Verification and storage. The invariant challenge_deadline_steps <
    // retained_checkpoints lives here: replaying a round needs the weights it
    // started from, and those exist only inside the retention window.
    uint32_t challenge_percent{};
    uint32_t challenge_deadline_steps{};
    uint32_t retained_checkpoints{};
    uint32_t slash_quorum{};
    uint8_t shadow_mode{};

    Status Validate() const;
    std::vector<uint8_t> Serialize() const;
    static Result<PolicyDescriptor> Deserialize(std::span<const uint8_t> content);
    std::vector<uint8_t> ToContainer() const;
    static Result<PolicyDescriptor> FromContainer(std::span<const uint8_t> raw);
    crypto::Hash256 Id() const;
};

// An order to recompute one contribution.
//
// The randomness that selects challenges must be unpredictable when a worker
// submits (otherwise it cheats only on rounds it knows are unchecked) and
// verifiable afterwards (otherwise a challenger fabricates targets). It therefore
// comes from an external beacon fixed AFTER submissions close, never from a value
// any participant here can grind.
struct ChallengeOrder {
    uint64_t round_id{};
    uint64_t outer_step{};
    crypto::Hash256 contribution_id{};
    crypto::Hash256 beacon{};        // external randomness, fixed after the round closed
    uint64_t beacon_height{};        // where it came from, so anyone can re-derive
    uint64_t verifier_id{};
    uint64_t issued_at_step{};       // starts the challenge deadline
    uint16_t determinism_class{};    // verifier and worker must share it

    Status Validate() const;
    std::vector<uint8_t> Serialize() const;
    static Result<ChallengeOrder> Deserialize(std::span<const uint8_t> content);
    std::vector<uint8_t> ToContainer() const;
    static Result<ChallengeOrder> FromContainer(std::span<const uint8_t> raw);
    crypto::Hash256 Id() const;
};

enum class VerdictKind : uint16_t {
    Match = 1,        // recomputation reproduced the claimed payload exactly
    Mismatch = 2,     // it did not — the contribution was not produced as claimed
    Unavailable = 3,  // inputs could not be obtained (pruned weights, missing payload)
    OutOfClass = 4,   // verifier's determinism class differs; it cannot judge this work
};

struct VerificationVerdict {
    crypto::Hash256 challenge_id{};
    crypto::Hash256 contribution_id{};
    uint64_t verifier_id{};
    VerdictKind kind{VerdictKind::Match};
    crypto::Hash256 claimed_payload_hash{};
    crypto::Hash256 recomputed_payload_hash{};
    uint16_t determinism_class{};
    uint64_t reported_at_step{};

    Status Validate() const;
    std::vector<uint8_t> Serialize() const;
    static Result<VerificationVerdict> Deserialize(std::span<const uint8_t> content);
    std::vector<uint8_t> ToContainer() const;
    static Result<VerificationVerdict> FromContainer(std::span<const uint8_t> raw);
    crypto::Hash256 Id() const;
};

// A quorum of mismatching verdicts. One verifier is not enough: a single
// colluding or faulty verifier must not be able to burn an honest worker's stake.
struct SlashEvidence {
    crypto::Hash256 contribution_id{};
    uint64_t worker_id{};
    uint64_t round_id{};
    std::vector<crypto::Hash256> verdict_ids;   // the mismatching verdicts, sorted
    uint16_t determinism_class{};
    uint64_t finalized_at_step{};

    Status Validate() const;
    std::vector<uint8_t> Serialize() const;
    static Result<SlashEvidence> Deserialize(std::span<const uint8_t> content);
    std::vector<uint8_t> ToContainer() const;
    static Result<SlashEvidence> FromContainer(std::span<const uint8_t> raw);
    crypto::Hash256 Id() const;
};

// The pre-image whose hash selects one training window. Serialized through canon
// so the derivation is defined by bytes, not by a language's struct layout —
// this is what lets a verifier reproduce a worker's batch exactly.
struct BatchSeed {
    crypto::Hash256 dataset_root{};
    uint64_t round_id{};
    uint64_t worker_id{};
    uint64_t step{};
    uint32_t index{};                   // which window within the micro-batch

    std::vector<uint8_t> Serialize() const;
    std::vector<uint8_t> ToContainer() const;
    crypto::Hash256 Id() const;
};

}  // namespace rnet::canon
