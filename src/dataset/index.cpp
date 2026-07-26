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

// Where chunk boundaries fall, given the corpus bytes.
//
// A chunk ends at the first document separator at or after `target_chunk_bytes`
// from its start; if none remains, it runs to the end of the corpus. Stated as a
// scan rather than a stored table so that every node holding the corpus derives
// the same boundaries from the same bytes, and the manifest stays small.
//
// The separator belongs to the chunk that precedes it. That is arbitrary but it
// has to be decided: a separator split across two chunks would make one of them
// begin with a blank line and the other end mid-separator, and two
// implementations would each pick a side.
Result<std::vector<uint64_t>> ChunkOffsets(const std::filesystem::path& corpus_file,
                                           uint32_t target_chunk_bytes, uint64_t n_bytes) {
    if (target_chunk_bytes == 0) return Err("index: target_chunk_bytes must be non-zero");
    if (n_bytes == 0) return Err("index: empty corpus");

    std::FILE* f = std::fopen(corpus_file.c_str(), "rb");
    if (!f) return Err("index: cannot open " + corpus_file.string());

    std::vector<uint64_t> offsets{0};
    // Read in windows that start where the search does, so a separator straddling
    // a read boundary is still found: the window always extends past the target.
    constexpr size_t kScan = 1u << 16;
    std::vector<uint8_t> buf(kScan);

    uint64_t cursor = 0;
    while (true) {
        const uint64_t search_from = cursor + target_chunk_bytes;
        if (search_from >= n_bytes) break;      // the remainder is the last chunk

        uint64_t at = search_from;
        uint64_t found = 0;
        while (at < n_bytes) {
            if (std::fseek(f, static_cast<long>(at), SEEK_SET) != 0) {
                std::fclose(f);
                return Err("index: seek failed while finding a chunk boundary");
            }
            const size_t want = static_cast<size_t>(std::min<uint64_t>(kScan, n_bytes - at));
            const size_t got = std::fread(buf.data(), 1, want, f);
            if (got != want) {
                std::fclose(f);
                return Err("index: short read while finding a chunk boundary");
            }
            // The separator is two bytes, so a match may span the window edge; the
            // next window starts one byte back to cover that.
            for (size_t i = 0; i + 1 < got; ++i) {
                if (buf[i] == '\n' && buf[i + 1] == '\n') {
                    found = at + i + 2;         // the separator ends the chunk
                    break;
                }
            }
            if (found != 0) break;
            if (at + got >= n_bytes) break;
            at += got - 1;
        }

        if (found == 0 || found >= n_bytes) break;   // no separator left: one final chunk
        offsets.push_back(found);
        cursor = found;
    }
    std::fclose(f);
    offsets.push_back(n_bytes);
    return offsets;
}

Result<DatasetIndex> DatasetIndex::Build(const std::filesystem::path& corpus_file,
                                         uint32_t target_chunk_bytes) {
    auto size = util::FileSize(corpus_file);
    if (!size) return Err(size.error());
    if (size.value() == 0) return Err("index: empty corpus");

    auto offsets = ChunkOffsets(corpus_file, target_chunk_bytes, size.value());
    if (!offsets) return Err(offsets.error());

    std::FILE* f = std::fopen(corpus_file.c_str(), "rb");
    if (!f) return Err("index: cannot open " + corpus_file.string());

    DatasetIndex idx;
    idx.n_bytes_ = size.value();
    idx.target_chunk_bytes_ = target_chunk_bytes;
    idx.offsets_ = std::move(offsets.value());

    std::vector<uint8_t> buffer;
    for (size_t i = 0; i + 1 < idx.offsets_.size(); ++i) {
        const uint64_t from = idx.offsets_[i];
        const uint64_t to = idx.offsets_[i + 1];
        if (to <= from) {
            std::fclose(f);
            return Err(std::format("index: chunk {} is empty ({}..{})", i, from, to));
        }
        buffer.resize(static_cast<size_t>(to - from));
        if (std::fseek(f, static_cast<long>(from), SEEK_SET) != 0 ||
            std::fread(buffer.data(), 1, buffer.size(), f) != buffer.size()) {
            std::fclose(f);
            return Err(std::format("index: short read for chunk {}", i));
        }
        idx.leaves_.push_back(crypto::MerkleLeaf(buffer));
    }
    std::fclose(f);

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
