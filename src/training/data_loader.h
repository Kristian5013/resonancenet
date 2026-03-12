#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

#include "core/error.h"

namespace rnet::training {

/// A batch of tokenized training data.
struct DataBatch {
    std::vector<int> tokens;    ///< [batch_size * seq_len] input tokens
    std::vector<int> targets;   ///< [batch_size * seq_len] target tokens (shifted by 1)
    int batch_size;
    int seq_len;
};

/// Loads tokenized training data and produces batches.
/// Data files are expected to be pre-tokenized binary (int32 per token).
class DataLoader {
public:
    /// Load a pre-tokenized dataset from disk.
    /// The file format is raw int32 tokens, little-endian.
    Result<void> load_dataset(const std::filesystem::path& path);

    /// Get the next batch. The batch contains input tokens and targets
    /// (shifted by 1 position for next-token prediction).
    DataBatch next_batch(int batch_size, int seq_len);

    /// Reset the position to the beginning of the dataset.
    void reset();

    /// Whether there are enough tokens remaining for at least one more batch.
    bool has_more() const;

    /// Total number of tokens in the loaded dataset.
    size_t total_tokens() const;

private:
    std::vector<int> tokens_;
    size_t pos_ = 0;
    int last_batch_size_ = 0;
    int last_seq_len_ = 0;
};

}  // namespace rnet::training
