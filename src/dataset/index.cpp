#include "dataset/index.h"

#include <algorithm>
#include <fcntl.h>
#include <unistd.h>

#include <cstdio>
#include <thread>
#include <cstring>
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

// Builds the tree in one sequential pass, hashing chunks in parallel.
//
// Three costs, and each one had to be dealt with before a 7.36 TB corpus could be
// indexed in less than a working day:
//
//   SEEKING. The first version found every boundary by seeking to
//   target_chunk_bytes past the last one and reading until a separator appeared.
//   Seven million seeks on a network-attached volume, each about a millisecond of
//   latency no throughput hides: measured at 111 MB/s, eighteen hours, and that
//   was before reading the file again to hash it.
//
//   SCANNING. Reading sequentially and testing every byte was 10.9 ns per byte —
//   92 MB/s, on a volume that delivers 1.2 GB/s to dd. Almost all of it was
//   wasted: a boundary cannot occur in the first target_chunk_bytes of a chunk, so
//   there is no reason to look there. The search now starts at the target and uses
//   memchr, which reduces it to roughly one scan per chunk.
//
//   HASHING. SHA3-256 runs at 334 MB/s per core here, so 7.36 TB is six hours on
//   one core however fast everything else is. Chunks are independent, so they are
//   hashed across every core, and the floor becomes the disk at 1.2 GB/s.
//
// The chunking rule is untouched by all of this, and had to be: it decides what
// chunk seven is. A chunk ends at the first document separator at or after
// target_chunk_bytes from its start, the separator belongs to the chunk it ends,
// and whatever remains at the end is the last chunk.
//   proven by: dataset_tests.cpp ChunksEndAtDocumentBoundaries
//   and cross-checked against the Python implementation, which still seeks:
//   worker/tests/test_cross_language.py test_chunking_agrees
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

    // Large enough to hold many chunks, so a batch is worth spreading over the
    // cores, and to absorb a chunk far bigger than the target — one enormous
    // document is a chunk of its own.
    const size_t window = std::max<size_t>(size_t{256} << 20,
                                           static_cast<size_t>(target_chunk_bytes) * 8);
    std::vector<uint8_t> buf(window);

    unsigned threads = std::thread::hardware_concurrency();
    if (threads == 0) threads = 4;

    size_t filled = 0;
    uint64_t window_start = 0;
    uint64_t chunk_start = 0;

    // Read through the page cache, but tell the kernel what is coming.
    //
    // Without this the reads went out one at a time — 256 KB requests at 1.43 ms
    // each, queue depth 1.38, which is 247 MB/s on a volume that gives dd 1.2 GB/s
    // with direct I/O. The disk was 72% idle and the process sat in
    // folio_wait_bit_common. Asking for the next window before hashing the current
    // one puts several requests in flight and overlaps the wait with the work.
    const int fd = ::fileno(f);
    ::posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL);

    while (true) {
        const size_t want = std::min<uint64_t>(buf.size() - filled, n_bytes - (window_start + filled));
        if (want > 0) {
            const uint64_t ahead = window_start + filled + want;
            if (ahead < n_bytes) {
                ::posix_fadvise(fd, static_cast<off_t>(ahead),
                                static_cast<off_t>(std::min<uint64_t>(buf.size(), n_bytes - ahead)),
                                POSIX_FADV_WILLNEED);
            }
            const size_t got = std::fread(buf.data() + filled, 1, want, f);
            if (got != want) {
                std::fclose(f);
                return Err(std::format("index: short read at byte {}", window_start + filled));
            }
            filled += got;
        }
        const bool at_eof = window_start + filled >= n_bytes;

        // Cut every chunk that ends inside this window.
        std::vector<std::pair<size_t, size_t>> batch;   // offsets relative to buf
        uint64_t cursor = chunk_start;
        while (true) {
            const uint64_t search_from = cursor + target_chunk_bytes;
            if (search_from + 1 >= window_start + filled) break;

            size_t rel = static_cast<size_t>(search_from - window_start);
            const uint8_t* p = buf.data() + rel;
            size_t remain = filled - rel;
            const uint8_t* hit = nullptr;
            while (remain >= 2) {
                const auto* nl = static_cast<const uint8_t*>(std::memchr(p, '\n', remain - 1));
                if (nl == nullptr) break;
                if (nl[1] == '\n') { hit = nl; break; }
                const size_t step = static_cast<size_t>(nl - p) + 1;
                p += step;
                remain -= step;
            }
            if (hit == nullptr) break;

            const uint64_t boundary = window_start + static_cast<size_t>(hit - buf.data()) + 2;
            if (boundary >= n_bytes) break;             // the tail is the final chunk
            batch.emplace_back(static_cast<size_t>(cursor - window_start),
                               static_cast<size_t>(boundary - window_start));
            idx.offsets_.push_back(boundary);
            cursor = boundary;
        }

        // The window just consumed will not be read again; letting it go keeps the
        // cache for what is coming rather than for 7 TB of history.
        ::posix_fadvise(fd, static_cast<off_t>(window_start), static_cast<off_t>(filled),
                        POSIX_FADV_DONTNEED);

        // Hash the batch across the cores. Leaves are written by index, so the
        // order is the chunk order however the threads interleave.
        if (!batch.empty()) {
            const size_t first = idx.leaves_.size();
            idx.leaves_.resize(first + batch.size());
            const unsigned n = std::min<unsigned>(threads, static_cast<unsigned>(batch.size()));
            std::vector<std::thread> pool;
            pool.reserve(n);
            for (unsigned t = 0; t < n; ++t) {
                pool.emplace_back([&, t] {
                    for (size_t i = t; i < batch.size(); i += n) {
                        idx.leaves_[first + i] = crypto::MerkleLeaf(std::span<const uint8_t>(
                            buf.data() + batch[i].first, batch[i].second - batch[i].first));
                    }
                });
            }
            for (auto& th : pool) th.join();
        }

        // The tail starts where the last boundary left the cursor, whether the
        // loop goes round again or stops here. Reading it from chunk_start
        // instead left the final chunk hashed from the previous window's
        // boundary, which failed its own inclusion proof and nothing else.
        chunk_start = cursor;
        if (at_eof) break;

        // Keep what the next chunk has already accumulated and refill behind it.
        const size_t keep_from = static_cast<size_t>(cursor - window_start);
        const size_t carried = filled - keep_from;
        if (carried > 0 && keep_from > 0) {
            std::memmove(buf.data(), buf.data() + keep_from, carried);
        }
        if (keep_from == 0 && filled == buf.size()) {
            // One chunk larger than the window: grow rather than stall.
            buf.resize(buf.size() * 2);
        }
        window_start = cursor;
        filled = carried;
    }
    std::fclose(f);

    // Whatever follows the last boundary is the final chunk, always.
    const size_t tail_from = static_cast<size_t>(chunk_start - window_start);
    idx.leaves_.push_back(crypto::MerkleLeaf(
        std::span<const uint8_t>(buf.data() + tail_from, filled - tail_from)));
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
