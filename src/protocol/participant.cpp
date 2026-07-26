#include "protocol/participant.h"

#include <algorithm>
#include <format>

#include "consensus/genesis.h"
#include "util/hex.h"
#include "util/logging.h"

namespace rnet::protocol {

std::string AdmissionName(Admission admission) {
    switch (admission) {
        case Admission::Accepted:  return "accepted";
        case Admission::Duplicate: return "duplicate";
        case Admission::NotForUs:  return "not-for-us";
        case Admission::Rejected:  return "rejected";
    }
    return "unknown";
}

namespace {

ObjectVerdict Reject(std::string reason, int misbehaviour) {
    return ObjectVerdict{Admission::Rejected, std::move(reason), misbehaviour};
}

ObjectVerdict Accept() { return ObjectVerdict{Admission::Accepted, {}, 0}; }

ObjectVerdict Duplicate() { return ObjectVerdict{Admission::Duplicate, "already known", 0}; }

ObjectVerdict NotForUs(std::string reason) {
    return ObjectVerdict{Admission::NotForUs, std::move(reason), 0};
}

// How many bytes a contribution's payload must be. Derived from the header, never
// from the sender: an int8 payload is exactly one byte per parameter, so a peer
// cannot make this node allocate by claiming a size.
uint64_t PayloadBytesFor(const canon::ContributionHeader& header) { return header.n_params; }

}  // namespace

Status ParticipantConfig::Validate() const {
    if (worker_id == 0) {
        // Worker 0 is the unset value. A node submitting under it would be
        // indistinguishable from a node that never configured itself, and the
        // round keys submissions by worker id.
        return Err("participant: worker_id must be set");
    }
    return Status::Ok();
}

Participant::Participant(ParticipantConfig config, canon::CheckpointHeader genesis_checkpoint)
    : config_(std::move(config)),
      params_(consensus::Params(config_.network)),
      chain_(params_.policy.retained_checkpoints),
      ledger_(params_.policy.slash_quorum, params_.policy.shadow_mode != 0) {
    if (auto st = chain_.SetGenesis(genesis_checkpoint); !st) {
        RNET_LOG_ERROR("participant: genesis checkpoint refused: {}", st.error());
        return;
    }
    base_checkpoint_ = genesis_checkpoint.Id();
    outer_step_ = genesis_checkpoint.outer_step;
    // The round is NOT opened here. Its deadline is measured from the moment it
    // opened, and a round opened at construction — before any clock is known —
    // would already be expired by the time the first real timestamp arrives, so
    // every submission would be refused as late.
}

Status Participant::OpenRound(uint64_t now_ms) {
    diloco::RoundConfig config;
    // Straight from the policy artifact. A node that used its own numbers here
    // would be running a different protocol while looking like a participant.
    config.inner_steps = params_.policy.inner_steps;
    config.min_contributors = params_.policy.min_contributors;
    // From the policy, not from this node. A round that never closes aggregates
    // nothing and issues no challenge, so the network would look alive and produce
    // nothing at all.
    config.deadline_ms = params_.policy.round_deadline_ms;
    if (auto st = config.Validate(); !st) return st;

    round_ = std::make_unique<diloco::Round>(params_.round.round_id, outer_step_, base_checkpoint_,
                                             now_ms, config);
    accepted_.clear();
    payloads_held_.clear();
    return Status::Ok();
}

// The beacon fixes who gets challenged, and it must not be knowable while a
// worker can still decide what to submit — otherwise a worker computes whether it
// will be checked before deciding whether to cheat. It is derived from the closed
// round's own evidence, which is not final until submissions are.
crypto::Hash256 Participant::BeaconFor(uint64_t outer_step) const {
    canon::Writer w;
    w.U64(params_.round.round_id);
    w.U64(outer_step);
    w.Hash(base_checkpoint_);
    w.Hash(round_ ? round_->EvidenceHash() : crypto::Hash256{});
    return crypto::Sha3_256(w.data());
}

ObjectVerdict Participant::OnObject(net::InvType type, const crypto::Hash256& hash,
                                    std::span<const uint8_t> container, uint64_t now_ms) {
    switch (type) {
        case net::InvType::Contribution:  return OnContribution(hash, container, now_ms);
        case net::InvType::Checkpoint:    return OnCheckpoint(hash, container);
        case net::InvType::Challenge:     return OnChallenge(hash, container);
        case net::InvType::Verdict:       return OnVerdict(hash, container);
        case net::InvType::SlashEvidence: return OnSlashEvidence(hash, container);
        case net::InvType::CorpusChunk:
            // Corpus bytes are checked against the dataset manifest by the
            // component that asked for them, not here.
            return NotForUs("corpus chunks are not consensus objects");
    }
    return Reject("unknown object type", 20);
}

ObjectVerdict Participant::OnContribution(const crypto::Hash256& id,
                                          std::span<const uint8_t> container, uint64_t now_ms) {
    auto header = canon::ContributionHeader::FromContainer(container);
    if (!header) return Reject("contribution does not decode: " + header.error(), 20);

    // The announced identity must be the object's own. A mismatch means the
    // announcement and the content disagree, and one of them is a lie.
    if (header.value().Id() != id) {
        return Reject("contribution id does not match its content", 50);
    }
    if (auto st = header.value().Validate(); !st) {
        return Reject("contribution is malformed: " + st.error(), 20);
    }
    if (accepted_.find(id) != accepted_.end()) return Duplicate();
    // Opened on first use, with the clock the caller supplied — and reopened if the
    // last one expired without a quorum. A contribution arriving at a node whose
    // round lapsed should start the next one, not be refused for the timing of a
    // tick it had no way to know about.
    if (!round_ || round_->StateAt(now_ms) == diloco::RoundState::Abandoned) {
        if (auto st = OpenRound(now_ms); !st) return Reject(st.error(), 0);
    }

    // Consensus checks. Each of these is a way a contribution can be well-formed
    // and still not belong to what this network is doing.
    if (header.value().round_id != params_.round.round_id) {
        return Reject(std::format("contribution is for round {}, this network runs round {}",
                                  header.value().round_id, params_.round.round_id),
                      20);
    }
    if (header.value().n_params != params_.round.model.ParameterCount()) {
        return Reject(std::format("contribution claims {} parameters, the model has {}",
                                  header.value().n_params, params_.round.model.ParameterCount()),
                      50);
    }
    // A worker that computed from weights this node does not have has not
    // contributed to this model, whatever its arithmetic was. Accepting it would
    // aggregate a delta against the wrong base.
    if (!chain_.Has(header.value().base_checkpoint)) {
        return Reject("contribution is based on a checkpoint this node does not hold", 10);
    }
    if (header.value().base_checkpoint != base_checkpoint_) {
        // Not misbehaviour: a slow worker submitting against the previous tip is
        // late, not dishonest.
        return NotForUs("contribution is based on a checkpoint that is no longer the tip");
    }
    if (header.value().inner_steps != params_.policy.inner_steps) {
        return Reject(std::format("contribution did {} inner steps, policy fixes {}",
                                  header.value().inner_steps, params_.policy.inner_steps),
                      20);
    }

    if (auto st = round_->Submit(header.value(), now_ms); !st) {
        // Round rules: one submission per worker, right step, before the deadline.
        // Breaking them is a protocol fault, not a race.
        return Reject(st.error(), 20);
    }
    accepted_[id] = header.value();
    return Accept();
}

ObjectVerdict Participant::OnCheckpoint(const crypto::Hash256& id,
                                        std::span<const uint8_t> container) {
    auto header = canon::CheckpointHeader::FromContainer(container);
    if (!header) return Reject("checkpoint does not decode: " + header.error(), 20);
    if (header.value().Id() != id) return Reject("checkpoint id does not match its content", 50);
    if (auto st = header.value().Validate(); !st) {
        return Reject("checkpoint is malformed: " + st.error(), 20);
    }
    if (chain_.Has(id)) return Duplicate();

    if (header.value().n_params != params_.round.model.ParameterCount()) {
        return Reject(std::format("checkpoint claims {} parameters, the model has {}",
                                  header.value().n_params, params_.round.model.ParameterCount()),
                      50);
    }
    // A checkpoint whose parent is unknown cannot be placed. Accepting it would
    // mean training onward from a state nobody proved — which is how a fork
    // becomes canonical without anyone deciding it should.
    if (!chain_.Has(header.value().parent)) {
        return NotForUs("checkpoint extends a parent this node does not hold yet");
    }
    // The producer's momentum, checked against this node's own rather than taken on
    // its word. This is the whole point of committing it: a checkpoint whose
    // optimizer state disagrees means the two nodes computed different updates from
    // the same contributions, and everything after it would compound the gap.
    if (optimizer_stepped_through_ > header.value().outer_step - 1 && !optimizer_desynced_) {
        if (header.value().optimizer_state_hash != optimizer_state_) {
            return Reject(
                std::format("checkpoint claims optimizer state {} but this node reached {} from "
                            "the same contributions",
                            util::ToHex(header.value().optimizer_state_hash).substr(0, 16),
                            util::ToHex(optimizer_state_).substr(0, 16)),
                50);
        }
    } else {
        // Cannot verify: this node never reproduced the aggregate for that step.
        // The chain may still be followed — the weights hash is there — but this
        // node knows it cannot produce until it catches up, and says so.
        optimizer_desynced_ = true;
        RNET_LOG_WARN(
            "participant: accepting checkpoint at step {} without verifying its optimizer state; "
            "this node cannot produce until it reproduces one",
            header.value().outer_step);
    }

    if (auto st = chain_.Append(header.value()); !st) {
        return Reject("checkpoint rejected by the chain: " + st.error(), 20);
    }

    auto tip = chain_.Tip();
    if (tip && tip.value().header.Id() == id) {
        // Eligibility for the NEXT step is settled here, by the round that just
        // ended. Drawing it from the round being closed instead would let a
        // producer omit whoever would have beaten it and elect itself.
        producer_pool_.clear();
        if (round_) {
            for (const auto& submitted : round_->OrderedSubmissions()) {
                producer_pool_.push_back(submitted.worker_id);
            }
        }
        // The tip moved: a new round opens against the new base. Contributions
        // still in flight against the old base become "late", not "invalid".
        base_checkpoint_ = id;
        outer_step_ = header.value().outer_step;
        // Dropped rather than reopened at an unknown time: the next round opens on
        // its first contribution or Tick, with a real clock.
        round_.reset();
        accepted_.clear();
        payloads_held_.clear();
        // The update was for the step just closed. The momentum inside the
        // optimizer carries forward — that is the point — but the staged update
        // does not, or the next step would publish the previous one's arithmetic.
        pending_update_.clear();
        pending_contributors_ = 0;
        RNET_LOG_INFO("participant: tip is now step {} ({})", outer_step_,
                      util::ToHex(id).substr(0, 16));
    }
    return Accept();
}

ObjectVerdict Participant::OnChallenge(const crypto::Hash256& id,
                                       std::span<const uint8_t> container) {
    auto order = canon::ChallengeOrder::FromContainer(container);
    if (!order) return Reject("challenge does not decode: " + order.error(), 20);
    if (order.value().Id() != id) return Reject("challenge id does not match its content", 50);
    if (auto st = order.value().Validate(); !st) {
        return Reject("challenge is malformed: " + st.error(), 20);
    }
    if (challenges_.find(id) != challenges_.end()) return Duplicate();

    if (order.value().round_id != params_.round.round_id) {
        return Reject("challenge is for another round", 20);
    }

    // The beacon decides who is challenged. Re-derived here rather than trusted:
    // without this check anyone could order anyone to burn a GPU-hour, which is
    // both a denial-of-service and a way to exhaust an honest worker.
    if (!verification::IsSelected(order.value().beacon, order.value().contribution_id,
                                  params_.policy.challenge_percent)) {
        return Reject("challenge names a contribution the beacon did not select", 50);
    }

    // The challenge must still be answerable. One issued against weights that have
    // already been discarded cannot be answered by anyone, and pretending
    // otherwise turns retention limits into unanswerable accusations.
    if (!verification::IsChallengeOpen(order.value(), outer_step_,
                                       params_.policy.challenge_deadline_steps)) {
        return NotForUs(std::format("challenge expired at step {}",
                                    verification::ChallengeExpiryStep(
                                        order.value(), params_.policy.challenge_deadline_steps)));
    }

    challenges_[id] = order.value();
    if (order.value().verifier_id != config_.worker_id) {
        // Kept, not discarded: this node needs to know the challenge exists in
        // order to judge the verdict that answers it.
        return NotForUs("challenge is assigned to another verifier");
    }
    if (!config_.verifier) return NotForUs("this node does not serve verification");
    return Accept();
}

ObjectVerdict Participant::OnVerdict(const crypto::Hash256& id,
                                     std::span<const uint8_t> container) {
    auto verdict = canon::VerificationVerdict::FromContainer(container);
    if (!verdict) return Reject("verdict does not decode: " + verdict.error(), 20);
    if (verdict.value().Id() != id) return Reject("verdict id does not match its content", 50);
    if (auto st = verdict.value().Validate(); !st) {
        return Reject("verdict is malformed: " + st.error(), 20);
    }

    // A verdict about nothing is how a ledger fills up with claims that have no
    // challenge to check them against.
    const auto challenge = challenges_.find(verdict.value().challenge_id);
    if (challenge == challenges_.end()) {
        return NotForUs("verdict answers a challenge this node has not seen");
    }
    // Only the assigned verifier may answer. Otherwise a worker could verify its
    // own contribution, or three colluding nodes could manufacture a quorum
    // against someone they were never asked to check.
    if (verdict.value().verifier_id != challenge->second.verifier_id) {
        return Reject("verdict is from a verifier this challenge was not assigned to", 50);
    }
    if (verdict.value().contribution_id != challenge->second.contribution_id) {
        return Reject("verdict answers a different contribution than its challenge", 50);
    }
    if (verdict.value().determinism_class != challenge->second.determinism_class) {
        // A verifier in another determinism class cannot produce a meaningful
        // bit-exact comparison, so its mismatch would be an artefact of hardware.
        return Reject("verdict comes from a different determinism class", 20);
    }

    if (auto st = ledger_.Record(verdict.value(), challenge->second); !st) {
        if (st.error().find("duplicate") != std::string::npos) return Duplicate();
        return Reject(st.error(), 20);
    }

    // Evidence is built and published; in shadow mode it settles nothing. The
    // economic consequence of a false accusation is permanent, so it waits until
    // the evidence has been produced by real hardware for long enough to believe.
    const auto outcome = ledger_.OutcomeFor(verdict.value().contribution_id);
    if (outcome == verification::EvidenceOutcome::Slashable) {
        // The worker being accused comes from the contribution this node accepted,
        // not from the verdict: a verifier naming a worker id would be choosing
        // who to accuse.
        const auto accused = accepted_.find(verdict.value().contribution_id);
        auto evidence = accused == accepted_.end()
                            ? Err("evidence: the accused contribution is not held")
                            : ledger_.BuildEvidence(verdict.value().contribution_id,
                                                    accused->second.worker_id,
                                                    params_.round.round_id, outer_step_);
        if (evidence) {
            outgoing_.push_back(Outgoing{net::InvType::SlashEvidence, evidence.value().Id(),
                                         evidence.value().ToContainer()});
            RNET_LOG_INFO("participant: slash evidence for {} ({}) — {}",
                          util::ToHex(verdict.value().contribution_id).substr(0, 16),
                          ledger_.shadow_mode() ? "shadow mode, nothing settled" : "SETTLING",
                          verification::OutcomeName(outcome));
        }
    }
    return Accept();
}

ObjectVerdict Participant::OnSlashEvidence(const crypto::Hash256& id,
                                           std::span<const uint8_t> container) {
    auto evidence = canon::SlashEvidence::FromContainer(container);
    if (!evidence) return Reject("evidence does not decode: " + evidence.error(), 20);
    if (evidence.value().Id() != id) return Reject("evidence id does not match its content", 50);
    if (auto st = evidence.value().Validate(); !st) {
        return Reject("evidence is malformed: " + st.error(), 20);
    }
    if (evidence.value().round_id != params_.round.round_id) {
        return Reject("evidence is for another round", 20);
    }
    // One verifier is not enough: a single colluding or faulty verifier must not
    // be able to burn an honest worker's stake. The quorum is a consensus value,
    // so it is checked against the policy rather than taken from the object.
    if (evidence.value().verdict_ids.size() < params_.policy.slash_quorum) {
        return Reject(std::format("evidence carries {} verdicts, quorum is {}",
                                  evidence.value().verdict_ids.size(),
                                  params_.policy.slash_quorum),
                      50);
    }
    // Recorded as a claim, never acted on here: this node reaches its own
    // conclusion from the verdicts it has verified itself.
    return Accept();
}

Status Participant::SubmitOwn(const canon::ContributionHeader& header, uint64_t now_ms) {
    if (header.worker_id != config_.worker_id) {
        return Err("participant: refusing to submit a contribution under another worker's id");
    }
    // Deliberately routed through the same path as a peer's. A node that accepts
    // from itself what it would refuse from a peer is a node whose own bugs never
    // surface until they reach the network.
    const auto container = header.ToContainer();
    const auto verdict = OnObject(net::InvType::Contribution, header.Id(), container, now_ms);
    if (!verdict.accepted()) {
        return Err("participant: own contribution refused: " + verdict.reason);
    }
    outgoing_.push_back(Outgoing{net::InvType::Contribution, header.Id(), container});
    // Our own payload is on disk already; nothing to fetch.
    payloads_held_.insert(header.payload_hash);
    return Status::Ok();
}

bool Participant::IsProducerForThisStep() const {
    // A node that cannot reproduce the chain's optimizer state is not a candidate,
    // whatever the election says. Producing from momentum nobody else has is the
    // failure this change exists to remove.
    if (optimizer_desynced_) return false;

    if (producer_pool_.empty()) {
        // Genesis has no previous evidence set, so the pool falls back to whoever
        // contributed to the round being closed. This is the one step where a
        // producer could in principle exclude a rival; it lasts exactly one step,
        // and from step 1 the pool is settled before the round opens.
        if (!round_) return false;
        std::vector<uint64_t> contributors;
        for (const auto& header : round_->OrderedSubmissions()) {
            contributors.push_back(header.worker_id);
        }
        return diloco::IsProducer(base_checkpoint_, outer_step_, contributors, config_.worker_id);
    }
    return diloco::IsProducer(base_checkpoint_, outer_step_, producer_pool_, config_.worker_id);
}

Status Participant::Tick(uint64_t now_ms) {
    // A round that has not opened has nothing to close. Opening it here keeps an
    // idle node's clock running so the first contribution is not already late.
    if (!round_) return OpenRound(now_ms);

    const auto state = round_->StateAt(now_ms);

    // A round whose deadline passed without a quorum is not an error: workers are
    // busy, slow or offline, and that is the normal state of an open network. What
    // would be an error is leaving it there — an abandoned round refuses every
    // later submission and never closes, so the node would go on ticking, logging
    // nothing, and never accept another contribution again. Silent permanent death.
    if (state == diloco::RoundState::Abandoned) {
        RNET_LOG_INFO("participant: round at step {} expired with {} of {} contributions; reopening",
                      outer_step_, round_->submission_count(),
                      params_.policy.min_contributors);
        return OpenRound(now_ms);
    }
    if (state != diloco::RoundState::Closed) return Status::Ok();

    // A CLOSED round is also a place to die in, and for longer than it took to
    // find that out for Abandoned.
    //
    // Closed has exactly two ways to end: this node produces the checkpoint, or
    // someone else's arrives and moves the tip. Both can simply not happen — a
    // payload that never lands, or an elected producer that crashed — and until
    // one does, `Round::Submit` refuses every new contribution because the round
    // is not Open. The node keeps ticking, logs nothing, and never accepts work
    // again.
    //
    // So a round that has been Closed for as long again as it was open has waited
    // long enough. One further deadline rather than a new consensus value: the
    // deadline already means "how long we wait for work", and waiting the same
    // for the aftermath is symmetric and adds nothing to re-pin.
    //
    // Reopening is safe for the follower case. A checkpoint that arrives late
    // still names the same parent, so OnCheckpoint accepts it and advances the tip
    // exactly as it would have.
    const uint64_t closed_at = round_->opened_ms() + params_.policy.round_deadline_ms;
    const bool waited_long_enough = now_ms > closed_at + params_.policy.round_deadline_ms;

    // Aggregation needs every accepted contribution's bytes. Producing from a
    // subset while claiming the whole set would give an evidence hash nobody could
    // reproduce.
    const auto missing = MissingPayloads();
    if (!missing.empty()) {
        if (!waited_long_enough) return Status::Ok();
        RNET_LOG_WARN(
            "participant: step {} closed with {} payload(s) that never arrived; abandoning them "
            "and reopening rather than waiting forever",
            outer_step_, missing.size());
        return OpenRound(now_ms);
    }

    if (auto st = IssueChallenges(round_->outer_step()); !st) return st;

    // EVERY node advances the optimizer here, producer or not.
    //
    // Momentum accumulates across steps and, with Nesterov, enters the next update
    // through the lookahead point rather than additively — so a node that steps
    // only when it happens to be elected arrives at its turn with momentum nobody
    // else has, and computes a different update from identical inputs. Measured
    // before this change: the same aggregate gave 56 on a node that had stepped
    // before and 33 on one that had not, and every test stayed green.
    if (auto st = AdvanceOptimizer(); !st) {
        RNET_LOG_ERROR("participant: cannot advance the optimizer at step {}: {}", outer_step_,
                       st.error());
        return Status::Ok();
    }

    // Exactly one node publishes; everyone else recomputes and compares. If each
    // node aggregated its own view, every node would produce a different checkpoint
    // for the same step and there would be no fact of the matter about which was
    // right.
    if (!IsProducerForThisStep()) {
        // Waiting for the producer's checkpoint is the normal state here. Waiting
        // for one that is never coming is not, and the two look identical from
        // inside this node.
        if (waited_long_enough) {
            RNET_LOG_WARN(
                "participant: step {} closed but the elected producer published nothing; "
                "reopening rather than waiting forever",
                outer_step_);
            return OpenRound(now_ms);
        }
        return Status::Ok();
    }
    return ProduceCheckpoint(now_ms);
}

Status Participant::AdvanceOptimizer() {
    // Once per outer step. Ticking twice while a round sits Closed must not step
    // twice — that is the same divergence arriving by a different door.
    if (optimizer_stepped_through_ > outer_step_) return Status::Ok();

    if (!payloads_) {
        // Without the payload bytes this node cannot reproduce the aggregate, so it
        // cannot reach the state the chain is about to be at. It may still follow;
        // it may not produce. Marked rather than ignored.
        optimizer_desynced_ = true;
        return Err("no payload provider");
    }
    const auto contributions = round_->OrderedSubmissions();
    if (contributions.empty()) return Status::Ok();

    // Borrowed, never copied: each payload is one byte per parameter, so a copy per
    // contribution would be a gigabyte per contributor.
    std::vector<diloco::Contribution> inputs;
    inputs.reserve(contributions.size());
    for (const auto& header : contributions) {
        const auto* bytes = payloads_(header.payload_hash);
        if (bytes == nullptr) {
            optimizer_desynced_ = true;
            return Err("a payload vanished between acceptance and aggregation");
        }
        if (bytes->size() != header.n_params) {
            optimizer_desynced_ = true;
            return Err("a payload is not one byte per parameter");
        }
        inputs.push_back(diloco::Contribution{
            header.worker_id, header.scale_exp,
            std::span<const int8_t>(reinterpret_cast<const int8_t*>(bytes->data()), bytes->size())});
    }

    auto aggregate = diloco::AverageContributions(inputs, params_.round.model.ParameterCount());
    if (!aggregate) {
        optimizer_desynced_ = true;
        return Err("aggregation failed: " + aggregate.error());
    }

    if (!optimizer_) {
        diloco::OuterOptimizerConfig config;
        config.momentum_q16 = params_.policy.outer_momentum_q16;
        config.outer_lr_q16 = params_.policy.outer_lr_q16;
        config.nesterov = params_.policy.nesterov != 0;
        optimizer_ = std::make_unique<diloco::OuterOptimizer>(
            params_.round.model.ParameterCount(), aggregate.value().scale_exp, config);
    }
    auto update = optimizer_->Step(aggregate.value());
    if (!update) {
        optimizer_desynced_ = true;
        return Err("outer step failed: " + update.error());
    }

    pending_update_ = std::move(update.value());
    pending_scale_exp_ = optimizer_->scale_exp();
    pending_evidence_ = diloco::EvidenceHashOf(contributions);
    pending_contributors_ = static_cast<uint32_t>(contributions.size());
    optimizer_state_ = optimizer_->StateHash();
    optimizer_stepped_through_ = outer_step_ + 1;
    optimizer_desynced_ = false;
    return Status::Ok();
}

Status Participant::ProduceCheckpoint(uint64_t now_ms) {
    (void)now_ms;
    // Already staged and waiting for the worker; producing again would discard the
    // update the worker may be halfway through applying.
    if (pending_) return Status::Ok();

    // A node that could not reach the chain's optimizer state must not publish one.
    // Its update would be computed from momentum nobody else has, and the network
    // would accept it because nothing else can check.
    if (optimizer_desynced_ || pending_update_.empty()) {
        RNET_LOG_WARN(
            "participant: elected to produce step {} but this node has not reproduced the "
            "optimizer state; refusing to publish an update nobody could verify",
            outer_step_ + 1);
        return Status::Ok();
    }


    auto tip = chain_.Tip();
    if (!tip) return Err("participant: no tip to extend");

    PendingProduction pending;
    pending.outer_step = outer_step_ + 1;
    pending.base_weights = tip.value().header.weights_hash;
    pending.scale_exp = pending_scale_exp_;
    pending.update = pending_update_;
    pending.evidence_hash = pending_evidence_;
    pending.contributor_count = pending_contributors_;
    pending_ = std::move(pending);
    RNET_LOG_INFO("participant: staged step {} from {} contributions, awaiting the worker",
                  pending_->outer_step, pending_->contributor_count);
    return Status::Ok();
}

Status Participant::CompleteProduction(const crypto::Hash256& weights_hash, uint64_t now_ms) {
    // A reply with nothing staged cannot be an answer to anything. Accepting it
    // would let a worker manufacture a checkpoint out of a bare hash.
    if (!pending_) return Err("participant: no checkpoint is waiting to be completed");
    if (weights_hash == crypto::Hash256{}) {
        return Err("participant: the worker reported an empty weights hash");
    }

    canon::CheckpointHeader header;
    header.parent = base_checkpoint_;
    header.round_id = params_.round.round_id;
    header.outer_step = pending_->outer_step;
    header.weights_hash = weights_hash;
    header.n_params = params_.round.model.ParameterCount();
    header.contributor_count = pending_->contributor_count;
    // What was aggregated, so anyone can fetch exactly these payloads, redo the
    // arithmetic and compare. Without it the checkpoint states an outcome and names
    // no inputs.
    header.evidence_hash = pending_->evidence_hash;
    // What this node's optimizer reached. A follower recomputes its own and
    // compares; without this the update is accepted on the producer's word.
    header.optimizer_state_hash = optimizer_state_;
    header.weight_dtype = 1;   // bf16
    pending_.reset();
    if (auto st = header.Validate(); !st) return Err(st.error());

    const auto container = header.ToContainer();
    const auto id = header.Id();
    // Routed through the ordinary path, so this node applies to its own checkpoint
    // every check it would apply to a peer's.
    const auto admission = OnObject(net::InvType::Checkpoint, id, container, now_ms);
    if (!admission.accepted()) {
        return Err("participant: own checkpoint refused: " + admission.reason);
    }
    outgoing_.push_back(Outgoing{net::InvType::Checkpoint, id, container});
    RNET_LOG_INFO("participant: produced checkpoint {} for step {} from {} contributions",
                  util::ToHex(id).substr(0, 16), header.outer_step, header.contributor_count);
    return Status::Ok();
}

Status Participant::IssueChallenges(uint64_t closed_step) {
    const auto contributions = round_->OrderedSubmissions();
    if (contributions.empty()) return Status::Ok();

    // Eligible verifiers are the workers who contributed in the class this round
    // runs in. A verifier in another class cannot produce a bit-exact comparison,
    // so assigning one would only manufacture false mismatches.
    std::vector<uint64_t> eligible;
    for (const auto& header : contributions) eligible.push_back(header.worker_id);
    std::sort(eligible.begin(), eligible.end());
    eligible.erase(std::unique(eligible.begin(), eligible.end()), eligible.end());
    if (eligible.size() < 2) {
        // With one worker there is nobody to check it who is not it.
        return Status::Ok();
    }

    const auto beacon = BeaconFor(closed_step);
    auto challenges = verification::SelectChallenges(
        contributions, beacon, /*beacon_height=*/closed_step, /*issued_at_step=*/closed_step,
        static_cast<uint16_t>(params_.round.determinism_class),
        eligible, params_.policy.challenge_percent);
    if (!challenges) return Err(challenges.error());

    for (const auto& order : challenges.value()) {
        const auto id = order.Id();
        if (challenges_.find(id) != challenges_.end()) continue;
        challenges_[id] = order;
        outgoing_.push_back(Outgoing{net::InvType::Challenge, id, order.ToContainer()});
    }
    return Status::Ok();
}

std::vector<Participant::MissingPayload> Participant::MissingPayloads() const {
    std::vector<MissingPayload> missing;
    for (const auto& [id, header] : accepted_) {
        if (payloads_held_.find(header.payload_hash) != payloads_held_.end()) continue;
        missing.push_back(MissingPayload{header.payload_hash, id, PayloadBytesFor(header)});
    }
    return missing;
}

Status Participant::NotePayload(const crypto::Hash256& payload_hash) {
    // The bytes were hash-verified by the net layer before reaching here, so this
    // records arrival rather than re-deciding validity. What it must not do is
    // record a payload nobody asked for: that would let a peer mark a
    // contribution complete without ever sending its data.
    const bool wanted = std::any_of(accepted_.begin(), accepted_.end(), [&](const auto& kv) {
        return kv.second.payload_hash == payload_hash;
    });
    if (!wanted) return Err("participant: no accepted contribution names this payload");
    payloads_held_.insert(payload_hash);
    return Status::Ok();
}

std::vector<canon::ChallengeOrder> Participant::PendingChallenges() const {
    std::vector<canon::ChallengeOrder> pending;
    if (!config_.verifier) return pending;
    for (const auto& [id, order] : challenges_) {
        if (order.verifier_id != config_.worker_id) continue;
        if (answered_.find(id) != answered_.end()) continue;
        if (!verification::IsChallengeOpen(order, outer_step_,
                                           params_.policy.challenge_deadline_steps)) {
            continue;   // expired: answering it would settle nothing
        }
        pending.push_back(order);
    }
    return pending;
}

Status Participant::SubmitVerdict(const canon::VerificationVerdict& verdict, uint64_t now_ms) {
    (void)now_ms;
    if (verdict.verifier_id != config_.worker_id) {
        return Err("participant: refusing to submit a verdict under another verifier's id");
    }
    const auto container = verdict.ToContainer();
    const auto admission = OnVerdict(verdict.Id(), container);
    if (!admission.accepted()) {
        return Err("participant: own verdict refused: " + admission.reason);
    }
    answered_.insert(verdict.challenge_id);
    outgoing_.push_back(Outgoing{net::InvType::Verdict, verdict.Id(), container});
    return Status::Ok();
}

std::vector<Outgoing> Participant::TakeOutgoing() { return std::exchange(outgoing_, {}); }

}  // namespace rnet::protocol
