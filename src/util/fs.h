// File I/O where every failure is reported.
//
// Reads verify the byte count; writes are atomic (temp file + rename) so a crash
// mid-write can never leave a half-written consensus artifact that later loads as
// "valid". Both properties are direct responses to defects found in the Lattica
// audit (unchecked fread, non-atomic checkpoint writes).
#pragma once

#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

#include "util/result.h"

namespace rnet::util {

Result<std::vector<uint8_t>> ReadFile(const std::filesystem::path& path);
Result<std::string> ReadTextFile(const std::filesystem::path& path);

// Writes to "<path>.tmp" then renames onto path. On success the file is complete;
// on failure the original (if any) is untouched.
Status WriteFileAtomic(const std::filesystem::path& path, std::span<const uint8_t> data);
Status WriteTextFileAtomic(const std::filesystem::path& path, std::string_view text);

Result<uint64_t> FileSize(const std::filesystem::path& path);

// Reads `len` bytes at `offset`. Short reads are an error, never silent zeros.
Result<std::vector<uint8_t>> ReadFileRange(const std::filesystem::path& path, uint64_t offset,
                                           size_t len);

}  // namespace rnet::util
