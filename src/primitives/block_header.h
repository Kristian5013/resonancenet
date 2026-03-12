#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "core/types.h"
#include "core/serialize.h"

namespace rnet::primitives {

/// CBlockHeader — block header for the Proof-of-Training blockchain.
/// Contains standard blockchain fields plus PoT-specific fields for
/// continuous model growth tracking.
struct CBlockHeader {
    // --- Core blockchain fields ---
    uint32_t version = 1;
    uint64_t height = 0;
    rnet::uint256 prev_hash{};
    rnet::uint256 merkle_root{};

    // --- Proof-of-Training fields ---
    rnet::uint256 checkpoint_hash{};       ///< Hash of the model checkpoint
    rnet::uint256 dataset_hash{};          ///< Hash of the training dataset
    float val_loss = 0.0f;                 ///< Validation loss achieved
    float prev_val_loss = 0.0f;            ///< Previous block's val_loss
    uint32_t train_steps = 0;              ///< Number of training steps performed

    // --- Model configuration at this height (CONTINUOUS GROWTH) ---
    uint32_t d_model = 384;                ///< Model dimension
    uint32_t n_layers = 6;                 ///< Number of transformer layers
    uint32_t n_slots = 64;                 ///< Number of attention slots
    uint32_t d_ff = 768;                   ///< Feed-forward dimension (= 2 * d_model)
    uint32_t vocab_size = 50257;           ///< Vocabulary size (GPT-2 default)
    uint32_t max_seq_len = 2048;           ///< Maximum sequence length
    uint8_t n_conv_branches = 5;           ///< Number of convolution branches
    std::array<uint8_t, 8> kernel_sizes = {3, 7, 15, 31, 63, 0, 0, 0};

    // --- Growth tracking ---
    uint32_t stagnation_count = 0;         ///< Consecutive blocks without improvement
    uint32_t growth_delta = 0;             ///< Growth event delta (0 = no growth)

    // --- Metadata ---
    uint64_t timestamp = 0;                ///< Block timestamp (seconds since epoch)
    std::array<uint8_t, 32> miner_pubkey{};  ///< Ed25519 public key of the miner
    std::array<uint8_t, 64> signature{};     ///< Ed25519 signature over unsigned header

    /// Compute the block hash: keccak256d of serialize_unsigned().
    rnet::uint256 hash() const;

    /// Serialize all fields EXCEPT the signature (for signing/hashing).
    std::vector<uint8_t> serialize_unsigned() const;

    /// Check if this is the genesis block (height == 0, prev_hash is zero).
    bool is_genesis() const {
        return height == 0 && prev_hash.is_zero();
    }

    /// Compute total model parameter count from current config.
    uint64_t model_param_count() const;

    /// Human-readable representation.
    std::string to_string() const;

    bool operator==(const CBlockHeader& other) const {
        return hash() == other.hash();
    }

    /// Full serialization (including signature) for network/disk.
    SERIALIZE_METHODS(
        READWRITE(self.version);
        READWRITE(self.height);
        READWRITE(self.prev_hash);
        READWRITE(self.merkle_root);
        READWRITE(self.checkpoint_hash);
        READWRITE(self.dataset_hash);
        READWRITE(self.val_loss);
        READWRITE(self.prev_val_loss);
        READWRITE(self.train_steps);
        READWRITE(self.d_model);
        READWRITE(self.n_layers);
        READWRITE(self.n_slots);
        READWRITE(self.d_ff);
        READWRITE(self.vocab_size);
        READWRITE(self.max_seq_len);
        READWRITE(self.n_conv_branches);
        READWRITE(self.kernel_sizes);
        READWRITE(self.stagnation_count);
        READWRITE(self.growth_delta);
        READWRITE(self.timestamp);
        READWRITE(self.miner_pubkey);
        READWRITE(self.signature);
    )
};

}  // namespace rnet::primitives
