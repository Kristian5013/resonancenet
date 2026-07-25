// Corpus integrity: a Merkle tree over fixed-size chunks of the token file.
//
// The root is a consensus value (it is pinned in the round descriptor), so a seed
// node cannot serve altered data without detection: any consumer verifies the
// chunk it received against the root using an inclusion proof.
//
// Chunking is by TOKEN COUNT, not bytes, so the tree shape is independent of the
// token width — a corpus re-encoded from uint16 to uint32 keeps the same chunk
// boundaries and therefore the same logical structure.
#pragma once

#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>

#include "canon/objects.h"
#include "crypto/merkle.h"
#include "util/result.h"

namespace rnet::dataset {

size_t TokenWidth(canon::TokenDtype dtype);

class DatasetIndex {
public:
    // Builds the tree by streaming the file; memory use is one chunk plus the leaf
    // hashes, so a multi-terabyte corpus indexes on a normal machine.
    static Result<DatasetIndex> Build(const std::filesystem::path& token_file,
                                      uint32_t chunk_tokens, canon::TokenDtype dtype);

    const crypto::Hash256& root() const { return root_; }
    uint64_t n_tokens() const { return n_tokens_; }
    uint32_t chunk_tokens() const { return chunk_tokens_; }
    uint32_t n_chunks() const { return static_cast<uint32_t>(leaves_.size()); }
    canon::TokenDtype dtype() const { return dtype_; }
    const std::vector<crypto::Hash256>& leaves() const { return leaves_; }

    canon::DatasetManifest ToManifest(const crypto::Hash256& tokenizer_hash) const;

    Result<crypto::MerkleProof> ProofForChunk(uint64_t chunk_index) const;

    // Which chunk holds a given token offset.
    Result<uint64_t> ChunkForToken(uint64_t token_offset) const;

private:
    crypto::Hash256 root_{};
    std::vector<crypto::Hash256> leaves_;
    uint64_t n_tokens_{0};
    uint32_t chunk_tokens_{0};
    canon::TokenDtype dtype_{canon::TokenDtype::Uint32};
};

// Verifies a served chunk against a manifest root without holding the tree.
bool VerifyChunk(std::span<const uint8_t> chunk_bytes, const crypto::MerkleProof& proof,
                 const crypto::Hash256& dataset_root);

}  // namespace rnet::dataset
