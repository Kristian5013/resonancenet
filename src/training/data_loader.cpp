// Copyright (c) 2024-2026 The ResonanceNet Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "training/data_loader.h"

#include "crypto/keccak.h"

#include <cstdio>
#include <cstring>

namespace rnet::training {

// ---------------------------------------------------------------------------
// load_dataset
// ---------------------------------------------------------------------------
// Loads tokenised training data from a binary file and computes the
// consensus dataset hash (Keccak-256d).
//
// The dataset hash pins the training data so that every full node can
// replay and verify the claimed validation loss.  Any change to the
// dataset produces a different hash, causing block rejection.
//
// Hash computation:
//   dataset_hash = Keccak256d(raw_file_bytes)
// ---------------------------------------------------------------------------
Result<void> DataLoader::load_dataset(const std::filesystem::path& path)
{
    // 1. Open the binary token file.
#ifdef _WIN32
    FILE* f = _wfopen(path.c_str(), L"rb");
#else
    FILE* f = std::fopen(path.string().c_str(), "rb");
#endif
    if (!f) {
        return Result<void>::err("Failed to open dataset file: " + path.string());
    }

    // 2. Determine file size and validate alignment.
    std::fseek(f, 0, SEEK_END);
    long file_size = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);

    if (file_size < 0 || file_size % sizeof(int32_t) != 0) {
        std::fclose(f);
        return Result<void>::err("Invalid dataset file size (must be multiple of 4 bytes)");
    }

    // 3. Compute token count and reject empty files.
    size_t n_tokens = static_cast<size_t>(file_size) / sizeof(int32_t);
    if (n_tokens == 0) {
        std::fclose(f);
        return Result<void>::err("Dataset file is empty");
    }

    tokens_.resize(n_tokens);

    // 4. Read in 4 MB chunks to keep memory pressure low on large datasets.
    constexpr size_t CHUNK_TOKENS = 1024 * 1024;  // 4 MB chunks
    size_t remaining = n_tokens;
    size_t offset = 0;
    while (remaining > 0) {
        size_t chunk = (remaining < CHUNK_TOKENS) ? remaining : CHUNK_TOKENS;
        size_t got = std::fread(tokens_.data() + offset, sizeof(int32_t), chunk, f);
        if (got != chunk) {
            std::fclose(f);
            return Result<void>::err("Failed to read dataset: truncated at offset " +
                                      std::to_string(offset));
        }
        offset += chunk;
        remaining -= chunk;
    }

    std::fclose(f);
    pos_ = 0;

    // 5. Compute consensus dataset hash over raw bytes.
    auto raw_bytes = std::span<const uint8_t>(
        reinterpret_cast<const uint8_t*>(tokens_.data()),
        n_tokens * sizeof(int32_t));
    dataset_hash_ = crypto::keccak256d(raw_bytes);

    return Result<void>::ok();
}

// ---------------------------------------------------------------------------
// next_batch
// ---------------------------------------------------------------------------
// Returns the next training batch of (tokens, targets) pairs.  Targets are
// the input sequence shifted by one position for next-token prediction.
// Wraps around to the beginning of the dataset when insufficient tokens
// remain.
// ---------------------------------------------------------------------------
DataBatch DataLoader::next_batch(int batch_size, int seq_len)
{
    DataBatch batch;
    batch.batch_size = batch_size;
    batch.seq_len = seq_len;
    last_batch_size_ = batch_size;
    last_seq_len_ = seq_len;

    // 1. Each sample needs seq_len + 1 tokens (the extra one is the final target).
    size_t tokens_per_sample = static_cast<size_t>(seq_len) + 1;

    // 2. Allocate output vectors.
    batch.tokens.resize(static_cast<size_t>(batch_size) * seq_len);
    batch.targets.resize(static_cast<size_t>(batch_size) * seq_len);

    // 3. Fill each sample in the batch.
    for (int b = 0; b < batch_size; ++b) {
        // Wrap around if we run out of data.
        if (pos_ + tokens_per_sample > tokens_.size()) {
            pos_ = 0;
        }

        size_t out_offset = static_cast<size_t>(b) * seq_len;
        for (int s = 0; s < seq_len; ++s) {
            batch.tokens[out_offset + s] = tokens_[pos_ + s];
            batch.targets[out_offset + s] = tokens_[pos_ + s + 1];  // shifted by 1
        }

        // 4. Advance by seq_len (not seq_len+1) to allow overlap.
        pos_ += seq_len;
    }

    return batch;
}

// ---------------------------------------------------------------------------
// reset
// ---------------------------------------------------------------------------
void DataLoader::reset()
{
    pos_ = 0;
}

// ---------------------------------------------------------------------------
// has_more
// ---------------------------------------------------------------------------
bool DataLoader::has_more() const
{
    if (tokens_.empty()) return false;

    // Need at least seq_len + 1 tokens per sample for one full batch.
    size_t needed = (last_seq_len_ > 0)
                    ? static_cast<size_t>(last_batch_size_) * (last_seq_len_ + 1)
                    : 2;
    return pos_ + needed <= tokens_.size();
}

// ---------------------------------------------------------------------------
// total_tokens
// ---------------------------------------------------------------------------
size_t DataLoader::total_tokens() const
{
    return tokens_.size();
}

// ---------------------------------------------------------------------------
// dataset_hash
// ---------------------------------------------------------------------------
rnet::uint256 DataLoader::dataset_hash() const
{
    return dataset_hash_;
}

} // namespace rnet::training
