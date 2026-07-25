// Serving the corpus.
//
// The deterministic batch derivation is the whole reason a worker cannot choose
// its own data: the window it reads is fixed by a hash of values it does not
// control. But that only holds if the worker actually reads THE corpus — the one
// the round descriptor pinned — and until a node can hand out chunks of it,
// nothing enforces that. A worker with a different file computes the same offsets
// into different tokens and nobody can tell.
//
// So chunks travel with an inclusion proof. A receiver checks the bytes against
// `dataset_root`, which is a consensus value, and a seed node that served altered
// data is caught by arithmetic rather than by reputation. That is what makes it
// safe for the corpus to come from a stranger.
//
// Two properties the design turns on:
//
//   * ADDRESSED BY INDEX, NOT BY HASH. A worker knows which chunk it needs — it
//     computed the offset itself — but not what that chunk hashes to. Asking by
//     hash would require it to already hold the thing it is asking for.
//
//   * PROOFS, NOT LEAF LISTS. A corpus of five trillion tokens is five million
//     chunks, and shipping every leaf hash up front is 160 MB before a worker has
//     read a single token. An inclusion proof is about 700 bytes against a four
//     megabyte chunk, and it is checked against a root the worker already trusts.
#pragma once

#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>

#include "canon/objects.h"
#include "crypto/merkle.h"
#include "dataset/index.h"
#include "util/result.h"

namespace rnet::net {

// One chunk is `chunk_tokens` wide — four megabytes on main — and no message
// carries that much, so a chunk crosses as several parts and is verified only
// once the whole of it has arrived. A part carries no proof of its own, exactly
// as a bulk chunk does not.
inline constexpr uint32_t kCorpusPartBytes = 512u * 1024;

// A node's copy of the corpus, plus the tree over it.
//
// Opening one costs the leaf hashes — 32 bytes per chunk — and nothing else: the
// token file itself is read a slice at a time, so a seed node serving a
// twenty-terabyte corpus holds megabytes, not terabytes.
class CorpusSource {
public:
    // Opens a token file and checks it against the manifest it claims to be. A
    // corpus whose root does not match is refused here rather than discovered by
    // the first worker that verifies a proof against it.
    static Result<CorpusSource> Open(const std::filesystem::path& token_file,
                                     const canon::DatasetManifest& manifest);

    const crypto::Hash256& root() const { return manifest_.dataset_root; }
    uint32_t n_chunks() const { return manifest_.n_chunks; }
    uint32_t chunk_tokens() const { return manifest_.chunk_tokens; }
    uint64_t n_tokens() const { return manifest_.n_tokens; }
    const canon::DatasetManifest& manifest() const { return manifest_; }

    // Bytes in a given chunk. The last one is short whenever the corpus does not
    // divide evenly, which is the normal case.
    Result<uint64_t> ChunkBytes(uint64_t chunk_index) const;

    // Reads a slice of one chunk. This is what the serving loop calls, so a
    // four-megabyte chunk is never resident just because a peer asked for it.
    Result<size_t> ReadChunkRange(uint64_t chunk_index, uint64_t offset,
                                  std::span<uint8_t> out) const;

    Result<crypto::MerkleProof> ProofForChunk(uint64_t chunk_index) const;

private:
    CorpusSource() = default;

    std::filesystem::path token_file_;
    canon::DatasetManifest manifest_{};
    dataset::DatasetIndex index_;
};

// Reassembles one corpus chunk and refuses anything that does not belong to the
// pinned root. A part is never trusted on its own; only the finished chunk is
// checked, and it is checked against consensus rather than against the sender.
class CorpusAssembler {
public:
    CorpusAssembler(uint64_t chunk_index, uint64_t total_bytes, crypto::MerkleProof proof);

    Status Accept(uint32_t part, std::span<const uint8_t> data);
    bool Complete() const;

    // Hands over the bytes only if they prove membership in `dataset_root`. A seed
    // node that served altered tokens fails here, and the worker learns it from
    // arithmetic rather than from trusting whoever answered.
    Result<std::vector<uint8_t>> Finish(const crypto::Hash256& dataset_root);

    uint64_t chunk_index() const { return chunk_index_; }
    uint32_t total_parts() const { return total_parts_; }

private:
    uint64_t chunk_index_;
    uint64_t total_bytes_;
    uint32_t total_parts_;
    uint32_t received_{0};
    crypto::MerkleProof proof_;
    std::vector<uint8_t> buffer_;
    std::vector<bool> have_;
};

}  // namespace rnet::net
