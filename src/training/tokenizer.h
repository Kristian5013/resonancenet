#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "core/error.h"

namespace rnet::training {

/// GPT-2 BPE tokenizer (50257 tokens).
/// Implements byte-level BPE encoding and decoding.
class Tokenizer {
public:
    static constexpr int VOCAB_SIZE = 50257;

    /// Special tokens — GPT-2 uses <|endoftext|> = 50256 for both PAD and EOS.
    static constexpr int PAD_TOKEN = 50256;
    static constexpr int EOS_TOKEN = 50256;

    /// Load vocabulary and merge rules from a directory containing
    /// vocab.json and merges.txt (GPT-2 format).
    Result<void> load(const std::filesystem::path& vocab_path);

    /// Encode text to token IDs using BPE.
    std::vector<int> encode(std::string_view text) const;

    /// Decode token IDs back to text.
    std::string decode(std::span<const int> tokens) const;

    /// Vocabulary size (always 50257 for GPT-2).
    int vocab_size() const { return VOCAB_SIZE; }

    /// Whether the tokenizer has been loaded.
    bool is_loaded() const { return loaded_; }

private:
    /// Apply BPE merges to a sequence of token strings.
    std::vector<std::string> bpe(const std::string& word) const;

    /// Token string -> ID mapping.
    std::unordered_map<std::string, int> encoder_;

    /// ID -> token string mapping.
    std::vector<std::string> decoder_;

    /// BPE merge pairs, ordered by priority.
    /// Key: (first, second) pair, Value: merge priority (lower = higher priority).
    std::unordered_map<std::string, int> bpe_ranks_;

    /// Byte-to-unicode mapping (GPT-2 style).
    std::array<std::string, 256> byte_encoder_;
    std::unordered_map<std::string, uint8_t> byte_decoder_;

    bool loaded_ = false;

    /// Initialize the byte-to-unicode mapping.
    void init_byte_mappings();
};

}  // namespace rnet::training
