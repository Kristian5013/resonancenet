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
Result<canon::DatasetManifest> BuildAndWriteManifest(const std::filesystem::path& corpus_file,
                                                     const std::filesystem::path& out_prefix,
                                                     uint32_t target_chunk_bytes,
                                                     const crypto::Hash256& tokenizer_hash);

Result<canon::DatasetManifest> LoadManifestFile(const std::filesystem::path& path);

// Re-indexes the corpus and confirms it reproduces the manifest's root, byte
// count and chunking. This is how a node checks a corpus it downloaded.
Status VerifyCorpus(const std::filesystem::path& corpus_file,
                    const canon::DatasetManifest& manifest);

util::Json ManifestToJson(const canon::DatasetManifest& m);

// Reads one chunk of the corpus, as the bytes it is.
//
// This does NOT return tokens. The node cannot tokenize — the BPE lives in the
// Python worker — and it does not need to: its part is to name which chunk a
// worker must train on and to serve those bytes with a proof. Turning them into
// tokens is the worker's, and reproducing that is the verifier's.
Result<std::vector<uint8_t>> ReadChunk(const std::filesystem::path& corpus_file,
                                       const canon::DatasetManifest& manifest,
                                       uint64_t chunk_index);

}  // namespace rnet::dataset
