#include "dataset/manifest.h"

#include <format>

#include "util/fs.h"
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
