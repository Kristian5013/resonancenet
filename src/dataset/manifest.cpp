#include "dataset/manifest.h"

#include <algorithm>
#include <format>

#include "util/fs.h"
#include "util/logging.h"
#include "util/hex.h"

namespace rnet::dataset {

Result<canon::DatasetManifest> BuildAndWriteManifest(const std::filesystem::path& corpus_file,
                                                     const std::filesystem::path& out_prefix,
                                                     uint32_t target_chunk_bytes,
                                                     const crypto::Hash256& tokenizer_hash) {
    auto index = DatasetIndex::Build(corpus_file, target_chunk_bytes);
    if (!index) return Err(index.error());

    const canon::DatasetManifest manifest = index.value().ToManifest(tokenizer_hash);
    if (auto st = manifest.Validate(); !st) return Err(st.error());

    const std::filesystem::path container_path = out_prefix.string() + ".rnds";
    if (auto st = util::WriteFileAtomic(container_path, manifest.ToContainer()); !st) {
        return Err(st.error());
    }

    const std::filesystem::path json_path = out_prefix.string() + ".json";
    if (auto st = util::WriteTextFileAtomic(json_path, ManifestToJson(manifest).Dump(2)); !st) {
        return Err(st.error());
    }

    // The index is the expensive half of publishing a corpus — reading all of it,
    // two and a half hours for 7.36 TB — and it has just been computed. Throwing
    // it away means the first node to serve that corpus pays the same cost again,
    // which is exactly what happened: dataset-build produced a root and discarded
    // the boundaries and leaves behind it, then a seed spent three hours
    // recomputing them before it could answer a single handshake.
    const std::filesystem::path cache_path = corpus_file.string() + ".rnidx";
    if (auto st = index.value().SaveCache(cache_path); !st) {
        // Not fatal: the manifest is what was asked for. The next reader rebuilds.
        RNET_LOG_WARN("corpus: cannot write the index cache {}: {}", cache_path.string(),
                      st.error());
    } else {
        RNET_LOG_INFO("corpus: index cached at {} — a node serving this corpus will not "
                      "have to read it again", cache_path.string());
    }
    return manifest;
}

Result<canon::DatasetManifest> LoadManifestFile(const std::filesystem::path& path) {
    auto raw = util::ReadFile(path);
    if (!raw) return Err(raw.error());
    return canon::DatasetManifest::FromContainer(raw.value());
}

Status VerifyCorpus(const std::filesystem::path& corpus_file,
                    const canon::DatasetManifest& manifest) {
    auto index = DatasetIndex::Build(corpus_file, manifest.target_chunk_bytes);
    if (!index) return Err(index.error());

    if (index.value().n_bytes() != manifest.n_bytes) {
        return Err(std::format("corpus: {} bytes on disk, manifest says {}",
                               index.value().n_bytes(), manifest.n_bytes));
    }
    if (index.value().n_chunks() != manifest.n_chunks) {
        return Err(std::format("corpus: {} chunks on disk, manifest says {}",
                               index.value().n_chunks(), manifest.n_chunks));
    }
    if (index.value().root() != manifest.dataset_root) {
        return Err("corpus: dataset_root mismatch — file does not match manifest\n  file:     " +
                   util::ToHex(index.value().root()) + "\n  manifest: " +
                   util::ToHex(manifest.dataset_root));
    }
    // The smallest chunk decides whether every assignment is trainable. A chunk
    // must hold seq_len + 1 tokens, and the rule gives the LAST chunk no minimum
    // at all — it is whatever follows the final boundary, which can be a single
    // byte. A worker sent there cannot form a window and fails on an assignment
    // nobody else ever draws, which is the worst shape a bug can have.
    uint64_t smallest = UINT64_MAX;
    uint64_t smallest_at = 0;
    const auto& offsets = index.value().offsets();
    for (size_t i = 0; i + 1 < offsets.size(); ++i) {
        const uint64_t width = offsets[i + 1] - offsets[i];
        if (width < smallest) { smallest = width; smallest_at = i; }
    }
    RNET_LOG_INFO("corpus: smallest chunk is {} bytes (chunk {}), largest is {} bytes",
                  smallest, smallest_at, [&] {
                      uint64_t big = 0;
                      for (size_t i = 0; i + 1 < offsets.size(); ++i) {
                          big = std::max(big, offsets[i + 1] - offsets[i]);
                      }
                      return big;
                  }());
    return Status::Ok();
}

util::Json ManifestToJson(const canon::DatasetManifest& m) {
    util::Json j = util::Json::Object();
    j.Set("dataset_root", util::ToHex(m.dataset_root));
    j.Set("tokenizer_hash", util::ToHex(m.tokenizer_hash));
    j.Set("n_bytes", m.n_bytes);
    j.Set("target_chunk_bytes", static_cast<int64_t>(m.target_chunk_bytes));
    j.Set("n_chunks", static_cast<int64_t>(m.n_chunks));

    j.Set("manifest_id", util::ToHex(m.Id()));
    return j;
}

Result<std::vector<uint8_t>> ReadChunk(const std::filesystem::path& corpus_file,
                                       const canon::DatasetManifest& manifest,
                                       uint64_t chunk_index) {
    // Boundaries are derived, not stored, so reading a chunk means deriving them
    // again. A node that serves chunks holds a DatasetIndex and does not come
    // through here; this is for one-off reads, where re-scanning is cheaper than
    // keeping a tree nobody else needs.
    auto index = DatasetIndex::Build(corpus_file, manifest.target_chunk_bytes);
    if (!index) return Err(index.error());
    if (index.value().root() != manifest.dataset_root) {
        return Err("chunk: the corpus on disk does not reproduce the manifest's root");
    }
    auto extent = index.value().ChunkExtent(chunk_index);
    if (!extent) return Err(extent.error());

    const auto [from, to] = extent.value();
    auto bytes = util::ReadFileRange(corpus_file, from, static_cast<size_t>(to - from));
    if (!bytes) return Err(bytes.error());
    return bytes.value();
}

}  // namespace rnet::dataset
