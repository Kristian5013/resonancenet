// Copyright (c) 2025 The ResonanceNet developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://opensource.org/licenses/MIT.

#include "primitives/block_header.h"

#include "core/stream.h"
#include "crypto/keccak.h"

namespace rnet::primitives {

// ===========================================================================
//  CBlockHeader
// ===========================================================================

// ---------------------------------------------------------------------------
// serialize_unsigned
//   Produces the canonical byte representation used for hashing.  The miner
//   signature is deliberately excluded so the hash commits to everything
//   the signature covers.
// ---------------------------------------------------------------------------
std::vector<uint8_t> CBlockHeader::serialize_unsigned() const
{
    core::DataStream ss;

    // 1. Serialize every field except signature
    core::Serialize(ss, version);
    core::Serialize(ss, height);
    ss << prev_hash;
    ss << merkle_root;
    ss << checkpoint_hash;
    ss << dataset_hash;
    core::Serialize(ss, val_loss);
    core::Serialize(ss, prev_val_loss);
    core::Serialize(ss, train_steps);
    core::Serialize(ss, d_model);
    core::Serialize(ss, n_layers);
    core::Serialize(ss, n_slots);
    core::Serialize(ss, d_ff);
    core::Serialize(ss, vocab_size);
    core::Serialize(ss, max_seq_len);
    core::Serialize(ss, n_conv_branches);
    core::Serialize(ss, kernel_sizes);
    core::Serialize(ss, stagnation_count);
    core::Serialize(ss, growth_delta);
    core::Serialize(ss, timestamp);
    core::Serialize(ss, miner_pubkey);

    return std::vector<uint8_t>(ss.data(), ss.data() + ss.size());
}

// ---------------------------------------------------------------------------
// hash
//   Keccak-256d of the unsigned serialisation.
// ---------------------------------------------------------------------------
rnet::uint256 CBlockHeader::hash() const
{
    auto data = serialize_unsigned();
    return crypto::keccak256d(std::span<const uint8_t>(data));
}

// ---------------------------------------------------------------------------
// model_param_count
//   Rough transformer parameter estimate:
//     Embedding:  vocab_size * d_model
//     Per-layer:  4*d_model^2 (attention) + 2*d_model*d_ff (FFN) + norms
//     Output:     d_model * vocab_size
// ---------------------------------------------------------------------------
uint64_t CBlockHeader::model_param_count() const
{
    uint64_t embed = static_cast<uint64_t>(vocab_size) * d_model;
    uint64_t per_layer = 4ULL * d_model * d_model +
                         2ULL * d_model * d_ff +
                         4ULL * d_model;  // layer norms
    uint64_t layers_total = static_cast<uint64_t>(n_layers) * per_layer;
    uint64_t output = static_cast<uint64_t>(d_model) * vocab_size;

    return embed + layers_total + output;
}

// ---------------------------------------------------------------------------
// to_string
// ---------------------------------------------------------------------------
std::string CBlockHeader::to_string() const
{
    std::string result = "CBlockHeader(";
    result += "ver=" + std::to_string(version);
    result += ", height=" + std::to_string(height);
    result += ", hash=" + hash().to_hex_rev().substr(0, 16) + "...";
    result += ", prev=" + prev_hash.to_hex_rev().substr(0, 16) + "...";
    result += ", val_loss=" + std::to_string(val_loss);
    result += ", d_model=" + std::to_string(d_model);
    result += ", n_layers=" + std::to_string(n_layers);
    result += ", time=" + std::to_string(timestamp);
    result += ")";
    return result;
}

} // namespace rnet::primitives
