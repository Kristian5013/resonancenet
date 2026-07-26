#include "dataset/index.h"

#include <algorithm>
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

// Builds the tree in ONE SEQUENTIAL PASS over the corpus.
//
// The obvious implementation is two passes: find the boundaries, then hash each
// chunk. It is also unusably slow. Finding a boundary means seeking to
// `target_chunk_bytes` past the last one and reading until a separator appears,
// and a 7.36 TB corpus at a 1 MiB target has seven million of those. On a
// network-attached volume each seek costs about a millisecond of latency that no
// amount of throughput hides: measured at 111 MB/s, which is eighteen hours, and
// that was before reading the whole file again to hash it.
//
// So the file is read once, front to back, in large blocks. Boundaries are found
// in the block already in hand, and each chunk is hashed incrementally as its
// bytes go by — MerkleLeaf is SHA3(0x00 || data), which streams.
//
// The rule is unchanged and must stay unchanged, because it decides what chunk
// seven is: a chunk ends at the first document separator at or after
// `target_chunk_bytes` from its start, the separator belongs to the chunk it
// ends, and if none remains the rest of the corpus is one final chunk.
//   proven by: dataset_tests.cpp ChunksEndAtDocumentBoundaries
Result<DatasetIndex> DatasetIndex::Build(const std::filesystem::path& corpus_file,
                                         uint32_t target_chunk_bytes) {
    if (target_chunk_bytes == 0) return Err("index: target_chunk_bytes must be non-zero");

    auto size = util::FileSize(corpus_file);
    if (!size) return Err(size.error());
    if (size.value() == 0) return Err("index: empty corpus");
    const uint64_t n_bytes = size.value();

    std::FILE* f = std::fopen(corpus_file.c_str(), "rb");
    if (!f) return Err("index: cannot open " + corpus_file.string());

    DatasetIndex idx;
    idx.n_bytes_ = n_bytes;
    idx.target_chunk_bytes_ = target_chunk_bytes;
    idx.offsets_.push_back(0);

    constexpr size_t kBlock = 8u << 20;
    std::vector<uint8_t> buf(kBlock);

    crypto::Sha3Hasher leaf;
    const uint8_t prefix = 0x00;                 // crypto::kLeafPrefix
    leaf.Update(std::span<const uint8_t>(&prefix, 1));

    uint64_t pos = 0;              // absolute position of buf[0]
    uint64_t chunk_start = 0;
    bool prev_was_newline = false; // a separator may straddle a block edge

    while (pos < n_bytes) {
        const size_t want = static_cast<size_t>(std::min<uint64_t>(kBlock, n_bytes - pos));
        const size_t got = std::fread(buf.data(), 1, want, f);
        if (got != want) {
            std::fclose(f);
            return Err(std::format("index: short read at byte {}", pos));
        }

        size_t consumed = 0;       // how much of this block is already hashed
        for (size_t i = 0; i < got; ++i) {
            const bool is_newline = buf[i] == '\n';
            const bool separator_ends_here = is_newline && prev_was_newline;
            prev_was_newline = is_newline && !separator_ends_here;

            if (!separator_ends_here) continue;
            const uint64_t boundary = pos + i + 1;          // one past the second \n
            if (boundary - chunk_start < target_chunk_bytes) continue;
            if (boundary >= n_bytes) break;                 // the tail is the last chunk

            leaf.Update(std::span<const uint8_t>(buf.data() + consumed, i + 1 - consumed));
            idx.leaves_.push_back(leaf.Finalize());
            idx.offsets_.push_back(boundary);
            consumed = i + 1;
            chunk_start = boundary;
            leaf = crypto::Sha3Hasher();
            leaf.Update(std::span<const uint8_t>(&prefix, 1));
        }

        if (consumed < got) {
            leaf.Update(std::span<const uint8_t>(buf.data() + consumed, got - consumed));
        }
        pos += got;
    }
    std::fclose(f);

    // Whatever remains after the last boundary is the final chunk, always.
    idx.leaves_.push_back(leaf.Finalize());
    idx.offsets_.push_back(n_bytes);

    idx.root_ = crypto::MerkleRoot(idx.leaves_);
    return idx;
}

Result<std::pair<uint64_t, uint64_t>> DatasetIndex::ChunkExtent(uint64_t chunk_index) const {
    if (chunk_index + 1 >= offsets_.size()) {
        return Err(std::format("index: chunk {} past the end ({} chunks)", chunk_index,
                               n_chunks()));
    }
    return std::pair<uint64_t, uint64_t>{offsets_[chunk_index], offsets_[chunk_index + 1]};
}

canon::DatasetManifest DatasetIndex::ToManifest(const crypto::Hash256& tokenizer_hash) const {
    canon::DatasetManifest m;
    m.dataset_root = root_;
    m.tokenizer_hash = tokenizer_hash;
    m.n_bytes = n_bytes_;
    m.target_chunk_bytes = target_chunk_bytes_;
    m.n_chunks = n_chunks();
    return m;
}

Result<crypto::MerkleProof> DatasetIndex::ProofForChunk(uint64_t chunk_index) const {
    return crypto::BuildMerkleProof(leaves_, chunk_index);
}


bool VerifyChunk(std::span<const uint8_t> chunk_bytes, const crypto::MerkleProof& proof,
                 const crypto::Hash256& dataset_root) {
    return crypto::VerifyMerkleProof(crypto::MerkleLeaf(chunk_bytes), proof, dataset_root);
}

}  // namespace rnet::dataset
