// Corpus integrity: a Merkle tree over document-aligned chunks of the text.
//
// The root is a consensus value (it is pinned in the round descriptor), so a seed
// node cannot serve altered data without detection: any consumer verifies the
// chunk it received against the root using an inclusion proof.
//
// The corpus is RAW UTF-8, not tokens, and a chunk ends at the first document
// separator at or after `target_chunk_bytes`. Two consequences the tree depends
// on:
//
//   * BOUNDARIES ARE DERIVED, NOT STORED. Any node holding the corpus recomputes
//     them by scanning, so the manifest carries three numbers rather than an
//     offset table that would be megabytes for a large corpus.
//
//   * EVERY CHUNK IS VALID UTF-8 AND WHOLE DOCUMENTS. A chunk cut at a fixed byte
//     boundary could begin mid-character, and then "what does this chunk say"
//     would have a different answer in C++ (bytes) and Python (str), on a path
//     nobody exercises. Cutting on separators removes the question.
//
// see: docs/corpus-addressing.md
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

// What ends a chunk: a blank line, which is how the corpus builder separates
// documents. Text rather than a reserved token id, because the corpus is not
// tokenized when it is written and there is no id to reserve.
inline constexpr std::string_view kDocumentSeparator = "\n\n";

class DatasetIndex {
public:
    // Builds the tree by streaming the file; memory use is one chunk plus the leaf
    // hashes, so a multi-terabyte corpus indexes on a normal machine.
    static Result<DatasetIndex> Build(const std::filesystem::path& corpus_file,
                                      uint32_t target_chunk_bytes);

    const crypto::Hash256& root() const { return root_; }
    uint64_t n_bytes() const { return n_bytes_; }
    uint32_t target_chunk_bytes() const { return target_chunk_bytes_; }
    uint32_t n_chunks() const { return static_cast<uint32_t>(leaves_.size()); }
    const std::vector<crypto::Hash256>& leaves() const { return leaves_; }

    // Where chunk `i` starts and ends. offsets_[n_chunks] is n_bytes, so a chunk's
    // extent is always offsets_[i+1] - offsets_[i].
    const std::vector<uint64_t>& offsets() const { return offsets_; }
    Result<std::pair<uint64_t, uint64_t>> ChunkExtent(uint64_t chunk_index) const;

    canon::DatasetManifest ToManifest(const crypto::Hash256& tokenizer_hash) const;

    // Writes the derived boundaries and leaf hashes beside the corpus, and reads
    // them back.
    //
    // Building the index means reading the whole corpus: 7.36 TB is two and a half
    // hours, and a seed paid it on every start, during which it answered nobody —
    // a node that cannot handshake for two hours looks exactly like a node that is
    // down. The cache is about 280 MB for that corpus and loads in seconds.
    //
    // What Load proves is that the cached leaves reproduce `expected_root`, which
    // is what makes them the corpus's commitment. What it does NOT prove is that
    // the file on disk still matches them — checking that is the whole scan again.
    // That is deliberate: a seed serving bytes that do not match is caught by the
    // receiver's inclusion proof, which is the guarantee the protocol actually
    // rests on, so re-reading terabytes to catch it earlier buys nothing.
    Status SaveCache(const std::filesystem::path& path) const;
    static Result<DatasetIndex> LoadCache(const std::filesystem::path& path,
                                          const crypto::Hash256& expected_root);

    Result<crypto::MerkleProof> ProofForChunk(uint64_t chunk_index) const;

private:
    crypto::Hash256 root_{};
    std::vector<crypto::Hash256> leaves_;
    std::vector<uint64_t> offsets_;      // n_chunks + 1 entries, ascending
    uint64_t n_bytes_{0};
    uint32_t target_chunk_bytes_{0};
};

// Verifies a served chunk against a manifest root without holding the tree.
bool VerifyChunk(std::span<const uint8_t> chunk_bytes, const crypto::MerkleProof& proof,
                 const crypto::Hash256& dataset_root);

}  // namespace rnet::dataset
