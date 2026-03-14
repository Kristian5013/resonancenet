#include "training/data_loader.h"

#include <cstdio>
#include <cstring>

#include "crypto/keccak.h"

namespace rnet::training {

Result<void> DataLoader::load_dataset(const std::filesystem::path& path) {
#ifdef _WIN32
    FILE* f = _wfopen(path.c_str(), L"rb");
#else
    FILE* f = std::fopen(path.string().c_str(), "rb");
#endif
    if (!f) {
        return Result<void>::err("Failed to open dataset file: " + path.string());
    }

    // Get file size
    std::fseek(f, 0, SEEK_END);
    long file_size = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);

    if (file_size < 0 || file_size % sizeof(int32_t) != 0) {
        std::fclose(f);
        return Result<void>::err("Invalid dataset file size (must be multiple of 4 bytes)");
    }

    size_t n_tokens = static_cast<size_t>(file_size) / sizeof(int32_t);
    if (n_tokens == 0) {
        std::fclose(f);
        return Result<void>::err("Dataset file is empty");
    }

    tokens_.resize(n_tokens);

    // Read in chunks for large files
    constexpr size_t CHUNK_TOKENS = 1024 * 1024;  // 4MB chunks
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

    // -----------------------------------------------------------------------
    // Compute Keccak-256d hash over the raw file bytes (the loaded tokens).
    // This ensures the dataset_hash reflects exactly what was loaded.
    // -----------------------------------------------------------------------
    auto raw_bytes = std::span<const uint8_t>(
        reinterpret_cast<const uint8_t*>(tokens_.data()),
        n_tokens * sizeof(int32_t));
    dataset_hash_ = crypto::keccak256d(raw_bytes);

    return Result<void>::ok();
}

DataBatch DataLoader::next_batch(int batch_size, int seq_len) {
    DataBatch batch;
    batch.batch_size = batch_size;
    batch.seq_len = seq_len;
    last_batch_size_ = batch_size;
    last_seq_len_ = seq_len;

    size_t tokens_per_sample = static_cast<size_t>(seq_len) + 1;  // +1 for target shift
    size_t total_needed = static_cast<size_t>(batch_size) * tokens_per_sample;

    batch.tokens.resize(static_cast<size_t>(batch_size) * seq_len);
    batch.targets.resize(static_cast<size_t>(batch_size) * seq_len);

    for (int b = 0; b < batch_size; ++b) {
        // Wrap around if we run out of data
        if (pos_ + tokens_per_sample > tokens_.size()) {
            pos_ = 0;
        }

        size_t out_offset = static_cast<size_t>(b) * seq_len;
        for (int s = 0; s < seq_len; ++s) {
            batch.tokens[out_offset + s] = tokens_[pos_ + s];
            batch.targets[out_offset + s] = tokens_[pos_ + s + 1];  // shifted by 1
        }

        pos_ += seq_len;  // advance by seq_len (not seq_len+1, to allow overlap)
    }

    return batch;
}

void DataLoader::reset() {
    pos_ = 0;
}

bool DataLoader::has_more() const {
    if (tokens_.empty()) return false;
    // Need at least seq_len + 1 tokens for one sample
    size_t needed = (last_seq_len_ > 0)
                    ? static_cast<size_t>(last_batch_size_) * (last_seq_len_ + 1)
                    : 2;
    return pos_ + needed <= tokens_.size();
}

size_t DataLoader::total_tokens() const {
    return tokens_.size();
}

rnet::uint256 DataLoader::dataset_hash() const {
    return dataset_hash_;
}

}  // namespace rnet::training
