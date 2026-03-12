#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "core/error.h"
#include "training/model_config.h"

namespace rnet::training {

/// .rnet checkpoint file format:
///   [4B magic "RNET"] [4B version] [ModelConfig serialized]
///   [8B step count] [8B tensor count]
///   For each tensor: [4B name_len][name][4B shape_dims][shape...][8B data_bytes][data...]
///   [32B keccak256d checksum of all preceding bytes]

struct CheckpointHeader {
    std::array<char, 4> magic = {'R', 'N', 'E', 'T'};
    uint32_t version = 1;
    ModelConfig config;
    uint64_t step = 0;
    uint64_t n_tensors = 0;
};

struct TensorEntry {
    std::string name;
    std::vector<int64_t> shape;
    std::vector<uint8_t> data;  // BF16 raw bytes
};

/// Read only the header from a .rnet checkpoint file.
Result<CheckpointHeader> read_checkpoint_header(const std::filesystem::path& path);

/// Read the full checkpoint (header + all tensors) from a .rnet file.
/// Verifies the trailing keccak256d checksum.
Result<std::vector<TensorEntry>> read_checkpoint(const std::filesystem::path& path);

/// Write a complete checkpoint file with trailing keccak256d checksum.
Result<void> write_checkpoint(const std::filesystem::path& path,
                               const CheckpointHeader& header,
                               const std::vector<TensorEntry>& tensors);

}  // namespace rnet::training
