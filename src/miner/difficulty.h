#pragma once

#include <cstdint>

#include "consensus/params.h"
#include "primitives/block_header.h"

namespace rnet::miner {

/// Difficulty helpers for Proof-of-Training.
///
/// There is no numeric hash target in PoT. "Difficulty" is the probability
/// of achieving val_loss < parent_val_loss within a given number of training
/// steps. These helpers estimate that probability and suggest step counts.

/// Estimate the probability of improving val_loss by at least the given
/// fraction after n_steps of training.
///
/// This is a heuristic based on empirical observations:
///   - Early training (high loss): improvement is very likely.
///   - Late training (low loss):   improvement becomes harder.
///
/// @param current_loss   The parent block's validation loss.
/// @param n_steps        Number of training steps to attempt.
/// @param d_model        Model dimension (larger = slower convergence).
/// @return               Estimated probability in [0.0, 1.0].
double estimate_improvement_probability(float current_loss,
                                        int n_steps,
                                        uint32_t d_model);

/// Suggest a reasonable step count for mining given the current state.
///
/// Balances:
///   - Higher steps → higher probability of improvement, but slower.
///   - Lower steps  → faster attempts, but more likely to fail.
///
/// @param current_loss   Parent block's validation loss.
/// @param d_model        Model dimension.
/// @param params         Consensus parameters (for min/max step bounds).
/// @return               Suggested step count.
int suggest_step_count(float current_loss,
                       uint32_t d_model,
                       const consensus::ConsensusParams& params);

/// Estimate the expected time (in seconds) for one mining attempt
/// given the step count and model size.
///
/// @param n_steps   Number of training steps.
/// @param d_model   Model dimension.
/// @param n_layers  Number of layers.
/// @return          Estimated wall-clock seconds.
double estimate_attempt_time(int n_steps,
                             uint32_t d_model,
                             uint32_t n_layers);

/// Compute the effective "difficulty" as a human-readable number.
/// This is the reciprocal of the estimated improvement probability.
/// Higher = harder to mine.
double effective_difficulty(float current_loss,
                            int n_steps,
                            uint32_t d_model);

}  // namespace rnet::miner
