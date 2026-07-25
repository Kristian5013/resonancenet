// Manifest lifecycle: build one from a corpus, persist it, verify a corpus
// against one, and read training windows out of it.
#pragma once

#include <cstdint>
#include <filesystem>
#include <vector>

#include "canon/objects.h"
#include "dataset/index.h"
#include "util/json.h"
#include "util/result.h"

namespace rnet::dataset {

// Builds a manifest for `token_file` and writes "<out_prefix>.rnds" (canonical
// container) plus "<out_prefix>.json" (the same facts, human-readable). The JSON
// is informational only — the container is authoritative.
Result<canon::DatasetManifest> BuildAndWriteManifest(const std::filesystem::path& token_file,
                                                     const std::filesystem::path& out_prefix,
                                                     uint32_t chunk_tokens,
                                                     canon::TokenDtype dtype,
                                                     const crypto::Hash256& tokenizer_hash);

Result<canon::DatasetManifest> LoadManifestFile(const std::filesystem::path& path);

// Re-indexes the corpus and confirms it reproduces the manifest's root, token
// count and chunking. This is how a node checks a corpus it downloaded.
Status VerifyCorpus(const std::filesystem::path& token_file, const canon::DatasetManifest& manifest);

util::Json ManifestToJson(const canon::DatasetManifest& m);

// Reads one training window (seq_len + 1 tokens, so the caller can form inputs
// and shifted targets) as 32-bit token ids, regardless of on-disk width.
Result<std::vector<uint32_t>> ReadWindow(const std::filesystem::path& token_file,
                                         const canon::DatasetManifest& manifest, uint64_t offset,
                                         uint32_t seq_len);

}  // namespace rnet::dataset
