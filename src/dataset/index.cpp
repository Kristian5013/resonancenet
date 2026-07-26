#include "dataset/index.h"

#include <cstdio>
#include <format>

#include "util/fs.h"

namespace rnet::dataset {

size_t TokenWidth(canon::TokenDtype dtype) {
    switch (dtype) {
        case canon::TokenDtype::Uint8:  return 1u;
        case canon::TokenDtype::Uint16: return 2u;
        case canon::TokenDtype::Uint32: return 4u;
    }
    return 4u;
}

Result<DatasetIndex> DatasetIndex::Build(const std::filesystem::path& token_file,
                                         uint32_t chunk_tokens, canon::TokenDtype dtype) {
    if (chunk_tokens == 0) return Err("index: chunk_tokens must be non-zero");
    if (!canon::IsKnownTokenDtype(dtype)) return Err("index: unknown token dtype");

    auto size = util::FileSize(token_file);
    if (!size) return Err(size.error());

    const size_t width = TokenWidth(dtype);
    if (size.value() % width != 0) {
        return Err(std::format("index: {} is not a whole number of {}-byte tokens",
                               token_file.string(), width));
    }
    const uint64_t n_tokens = size.value() / width;
    if (n_tokens == 0) return Err("index: empty corpus");

    std::FILE* f = std::fopen(token_file.c_str(), "rb");
    if (!f) return Err("index: cannot open " + token_file.string());

    DatasetIndex idx;
    idx.n_tokens_ = n_tokens;
    idx.chunk_tokens_ = chunk_tokens;
    idx.dtype_ = dtype;

    const size_t chunk_bytes = static_cast<size_t>(chunk_tokens) * width;
    std::vector<uint8_t> buffer(chunk_bytes);
    uint64_t remaining = size.value();

    while (remaining > 0) {
        const size_t want = static_cast<size_t>(std::min<uint64_t>(remaining, chunk_bytes));
        const size_t got = std::fread(buffer.data(), 1, want, f);
        if (got != want) {
            std::fclose(f);
            return Err(std::format("index: short read at {} bytes remaining", remaining));
        }
        idx.leaves_.push_back(crypto::MerkleLeaf(std::span<const uint8_t>(buffer.data(), want)));
        remaining -= want;
    }
    std::fclose(f);

    idx.root_ = crypto::MerkleRoot(idx.leaves_);
    return idx;
}

canon::DatasetManifest DatasetIndex::ToManifest(const crypto::Hash256& tokenizer_hash) const {
    canon::DatasetManifest m;
    m.dataset_root = root_;
    m.tokenizer_hash = tokenizer_hash;
    m.n_tokens = n_tokens_;
    m.chunk_tokens = chunk_tokens_;
    m.n_chunks = n_chunks();
    m.dtype = dtype_;
    return m;
}

Result<crypto::MerkleProof> DatasetIndex::ProofForChunk(uint64_t chunk_index) const {
    return crypto::BuildMerkleProof(leaves_, chunk_index);
}

Result<uint64_t> DatasetIndex::ChunkForToken(uint64_t token_offset) const {
    if (token_offset >= n_tokens_) return Err("index: token offset past end of corpus");
    return token_offset / chunk_tokens_;
}

bool VerifyChunk(std::span<const uint8_t> chunk_bytes, const crypto::MerkleProof& proof,
                 const crypto::Hash256& dataset_root) {
    return crypto::VerifyMerkleProof(crypto::MerkleLeaf(chunk_bytes), proof, dataset_root);
}

}  // namespace rnet::dataset
