#include "dataset/manifest.h"

#include <format>

#include "util/fs.h"
#include "util/hex.h"

namespace rnet::dataset {

Result<canon::DatasetManifest> BuildAndWriteManifest(const std::filesystem::path& token_file,
                                                     const std::filesystem::path& out_prefix,
                                                     uint32_t chunk_tokens,
                                                     canon::TokenDtype dtype,
                                                     const crypto::Hash256& tokenizer_hash) {
    auto index = DatasetIndex::Build(token_file, chunk_tokens, dtype);
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

Status VerifyCorpus(const std::filesystem::path& token_file,
                    const canon::DatasetManifest& manifest) {
    auto index = DatasetIndex::Build(token_file, manifest.chunk_tokens, manifest.dtype);
    if (!index) return Err(index.error());

    if (index.value().n_tokens() != manifest.n_tokens) {
        return Err(std::format("corpus: {} tokens on disk, manifest says {}",
                               index.value().n_tokens(), manifest.n_tokens));
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
    j.Set("n_tokens", m.n_tokens);
    j.Set("chunk_tokens", static_cast<int64_t>(m.chunk_tokens));
    j.Set("n_chunks", static_cast<int64_t>(m.n_chunks));
    j.Set("dtype", m.dtype == canon::TokenDtype::Uint16 ? "uint16" : "uint32");
    j.Set("manifest_id", util::ToHex(m.Id()));
    return j;
}

Result<std::vector<uint32_t>> ReadWindow(const std::filesystem::path& token_file,
                                         const canon::DatasetManifest& manifest, uint64_t offset,
                                         uint32_t seq_len) {
    const uint64_t need = static_cast<uint64_t>(seq_len) + 1;   // inputs + shifted target
    if (offset + need > manifest.n_tokens) {
        return Err(std::format("window: offset {} + {} exceeds corpus ({} tokens)", offset, need,
                               manifest.n_tokens));
    }
    const size_t width = TokenWidth(manifest.dtype);
    auto bytes = util::ReadFileRange(token_file, offset * width, static_cast<size_t>(need) * width);
    if (!bytes) return Err(bytes.error());

    std::vector<uint32_t> tokens(static_cast<size_t>(need));
    const auto& b = bytes.value();
    // Token payloads are little-endian on disk: they are produced by the worker's
    // numpy pipeline, and this is the one place the protocol adopts LE rather than
    // the big-endian convention used for metadata.
    if (manifest.dtype == canon::TokenDtype::Uint16) {
        for (size_t i = 0; i < tokens.size(); ++i) {
            tokens[i] = static_cast<uint32_t>(b[2 * i]) | (static_cast<uint32_t>(b[2 * i + 1]) << 8);
        }
    } else {
        for (size_t i = 0; i < tokens.size(); ++i) {
            tokens[i] = static_cast<uint32_t>(b[4 * i]) |
                        (static_cast<uint32_t>(b[4 * i + 1]) << 8) |
                        (static_cast<uint32_t>(b[4 * i + 2]) << 16) |
                        (static_cast<uint32_t>(b[4 * i + 3]) << 24);
        }
    }
    return tokens;
}

}  // namespace rnet::dataset
