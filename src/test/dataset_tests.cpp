#include <cstdio>
#include <filesystem>
#include <string>

#include "dataset/index.h"
#include "dataset/manifest.h"
#include "dataset/scheduler.h"
#include "test/framework.h"
#include "util/fs.h"
#include "util/hex.h"

using namespace rnet;
using namespace rnet::dataset;

namespace {

// A scratch corpus of UTF-8 text: `n_docs` documents separated by a blank line,
// which is how the corpus builder writes one.
//
// Document lengths vary on purpose. Equal-length documents put chunk boundaries
// at regular intervals, which is exactly the case that would hide a bug in
// deriving them.
std::filesystem::path WriteCorpus(const std::string& name, size_t n_docs, size_t base = 400) {
    const auto path = std::filesystem::temp_directory_path() / ("rnet_test_" + name + ".txt");
    std::string text;
    for (size_t i = 0; i < n_docs; ++i) {
        const size_t len = base + (i * 137) % (base * 3);
        text.append("Document ").append(std::to_string(i)).append(". ");
        for (size_t j = 0; j < len; ++j) {
            text.push_back(static_cast<char>('a' + ((i * 31 + j * 7) % 26)));
            if (j % 9 == 8) text.push_back(' ');
        }
        text.append("\n\n");
    }
    (void)util::WriteTextFileAtomic(path, text);
    return path;
}

const crypto::Hash256 kRoot = crypto::Sha3_256(std::string_view("test-dataset-root"));

crypto::Hash256 Seed(uint64_t worker, uint64_t step, uint32_t index) {
    return WindowSeed(kRoot, 0, worker, step, index);
}

}  // namespace

// ---------------------------------------------------------------------------
// Deterministic batch derivation — the property that closes data poisoning.
// ---------------------------------------------------------------------------

RNET_TEST(Schedule, ChunksAreDeterministic) {
    auto a = BatchChunks(kRoot, 0, 7, 42, 4, 1'000);
    auto b = BatchChunks(kRoot, 0, 7, 42, 4, 1'000);
    RNET_CHECK_OK(a);
    RNET_CHECK_OK(b);
    RNET_CHECK_EQ(a.value(), b.value());
}

RNET_TEST(Schedule, DifferentWorkersGetDifferentData) {
    auto w1 = BatchChunks(kRoot, 0, 1, 5, 8, 100'000);
    auto w2 = BatchChunks(kRoot, 0, 2, 5, 8, 100'000);
    RNET_CHECK_OK(w1);
    RNET_CHECK_OK(w2);
    RNET_CHECK(w1.value() != w2.value());
}

// Every input must move the result. An input that does not is an input a worker
// could vary freely to fish for data it likes.
RNET_TEST(Schedule, EveryInputAffectsTheSeed) {
    const auto base = WindowSeed(kRoot, 3, 9, 100, 0);
    const auto other_root = crypto::Sha3_256(std::string_view("another corpus"));
    for (const auto& v : {WindowSeed(other_root, 3, 9, 100, 0),
                          WindowSeed(kRoot, 4, 9, 100, 0),
                          WindowSeed(kRoot, 3, 10, 100, 0),
                          WindowSeed(kRoot, 3, 9, 101, 0),
                          WindowSeed(kRoot, 3, 9, 100, 1)}) {
        RNET_CHECK(v != base);
    }
}

RNET_TEST(Schedule, ChunksStayInBounds) {
    for (uint64_t n_chunks : {uint64_t{1}, uint64_t{2}, uint64_t{97}, uint64_t{1'000'000}}) {
        for (uint32_t i = 0; i < 64; ++i) {
            auto c = ChunkForWindow(Seed(i, i * 3, i), n_chunks);
            RNET_CHECK_OK(c);
            if (c.value() >= n_chunks) {
                RNET_FAIL("chunk {} is outside a corpus of {}", c.value(), n_chunks);
            }
        }
    }
}

// The offset within a chunk comes from a different hash than the chunk index, so
// knowing one tells a worker nothing about the other.
RNET_TEST(Schedule, TheOffsetIsNotTheChunkDrawReused) {
    const auto seed = Seed(11, 22, 3);
    auto chunk = ChunkForWindow(seed, 1'000'000);
    auto offset = OffsetInChunk(seed, 1'000'064, 64);
    RNET_CHECK_OK(chunk);
    RNET_CHECK_OK(offset);
    RNET_CHECK(chunk.value() != offset.value());
}

RNET_TEST(Schedule, OffsetsStayInBounds) {
    const uint32_t seq = 64;
    for (uint64_t tokens : {uint64_t{65}, uint64_t{100}, uint64_t{200'000}}) {
        for (uint32_t i = 0; i < 64; ++i) {
            auto off = OffsetInChunk(Seed(i, i, i), tokens, seq);
            RNET_CHECK_OK(off);
            // seq_len inputs plus one shifted target.
            if (off.value() + seq + 1 > tokens) {
                RNET_FAIL("offset {} + {} exceeds a chunk of {} tokens", off.value(), seq + 1,
                          tokens);
            }
        }
    }
}

// A chunk too small for one window is refused rather than clamped. The manifest
// is meant to make this impossible when the corpus is built, so arriving here
// means something upstream is wrong, and saying so beats quietly training on a
// short window.
RNET_TEST(Schedule, AChunkTooSmallForAWindowIsAnError) {
    RNET_CHECK_ERR(OffsetInChunk(Seed(1, 1, 1), 64, 64));
    RNET_CHECK_OK(OffsetInChunk(Seed(1, 1, 1), 65, 64));
    RNET_CHECK_ERR(ChunkForWindow(Seed(1, 1, 1), 0));
}

// The values the Python mirror is asserted against by
// worker/tests/test_cross_language.py, which drives the real rnet-tool binary.
// Printed rather than compared to a literal here: this file agreeing with itself
// proves nothing about the other implementation.
RNET_TEST(Schedule, GoldenVector) {
    const auto seed = WindowSeed(kRoot, 1, 2, 3, 0);
    auto chunk = ChunkForWindow(seed, 100'000);
    auto offset = OffsetInChunk(seed, 200'000, 8192);
    RNET_CHECK_OK(chunk);
    RNET_CHECK_OK(offset);
    std::printf("        [golden] seed %s chunk %llu offset %llu\n",
                util::ToHex(seed).substr(0, 16).c_str(),
                static_cast<unsigned long long>(chunk.value()),
                static_cast<unsigned long long>(offset.value()));
    RNET_CHECK(chunk.value() < 100'000);
    RNET_CHECK(offset.value() + 8193 <= 200'000);
}

// ---------------------------------------------------------------------------
// Chunking: derived from the bytes, aligned to documents.
// ---------------------------------------------------------------------------

RNET_TEST(Index, ChunksEndAtDocumentBoundaries) {
    const auto path = WriteCorpus("aligned", 200);
    auto raw = util::ReadFile(path);
    RNET_CHECK_OK(raw);
    const auto& bytes = raw.value();

    auto index = DatasetIndex::Build(path, 4096);
    RNET_CHECK_OK(index);
    const auto& offsets = index.value().offsets();

    RNET_CHECK_EQ(offsets.front(), uint64_t{0});
    RNET_CHECK_EQ(offsets.back(), index.value().n_bytes());
    RNET_CHECK(offsets.size() > 2);

    for (size_t i = 1; i + 1 < offsets.size(); ++i) {
        const uint64_t at = offsets[i];
        // Every interior boundary sits just past a blank line, so the two bytes
        // before it are the separator. A boundary anywhere else cuts a document,
        // and could cut a UTF-8 character with it.
        if (at < 2 || bytes[at - 1] != '\n' || bytes[at - 2] != '\n') {
            RNET_FAIL("boundary {} at byte {} does not follow a blank line", i, at);
        }
        if (at - offsets[i - 1] < 4096) {
            RNET_FAIL("chunk {} is {} bytes, shorter than the 4096-byte target", i - 1,
                      at - offsets[i - 1]);
        }
    }
    std::filesystem::remove(path);
}

RNET_TEST(Index, BoundariesAreTheSameOnEveryNode) {
    const auto path = WriteCorpus("stable", 150);
    auto first = DatasetIndex::Build(path, 4096);
    auto second = DatasetIndex::Build(path, 4096);
    RNET_CHECK_OK(first);
    RNET_CHECK_OK(second);
    RNET_CHECK_EQ(first.value().offsets(), second.value().offsets());
    RNET_CHECK_EQ(first.value().root(), second.value().root());

    // A different target is a different corpus as far as the root is concerned,
    // which is why target_chunk_bytes is in the manifest and not a local setting.
    auto coarser = DatasetIndex::Build(path, 16384);
    RNET_CHECK_OK(coarser);
    RNET_CHECK(coarser.value().root() != first.value().root());
    std::filesystem::remove(path);
}

RNET_TEST(Index, BuildsAndVerifies) {
    const auto path = WriteCorpus("verify", 120);
    auto index = DatasetIndex::Build(path, 4096);
    RNET_CHECK_OK(index);
    RNET_CHECK(index.value().n_chunks() > 1);

    const auto manifest = index.value().ToManifest(crypto::Hash256{});
    RNET_CHECK_OK(manifest.Validate());
    RNET_CHECK_OK(VerifyCorpus(path, manifest));
    std::filesystem::remove(path);
}

RNET_TEST(Index, AlteredCorpusFailsVerification) {
    const auto path = WriteCorpus("altered", 120);
    auto index = DatasetIndex::Build(path, 4096);
    RNET_CHECK_OK(index);
    const auto manifest = index.value().ToManifest(crypto::Hash256{});

    auto raw = util::ReadFile(path);
    RNET_CHECK_OK(raw);
    auto tampered = raw.value();
    tampered[tampered.size() / 2] ^= 0x01;      // one bit, in the middle
    RNET_CHECK_OK(util::WriteFileAtomic(path, tampered));

    RNET_CHECK_ERR(VerifyCorpus(path, manifest));
    std::filesystem::remove(path);
}

RNET_TEST(Index, ChunkProofsVerifyAgainstRoot) {
    const auto path = WriteCorpus("proofs", 160);
    auto index = DatasetIndex::Build(path, 4096);
    RNET_CHECK_OK(index);
    auto raw = util::ReadFile(path);
    RNET_CHECK_OK(raw);

    for (uint64_t i = 0; i < index.value().n_chunks(); ++i) {
        auto extent = index.value().ChunkExtent(i);
        RNET_CHECK_OK(extent);
        const uint64_t from = extent.value().first;
        const uint64_t to = extent.value().second;
        std::span<const uint8_t> chunk(raw.value().data() + from, static_cast<size_t>(to - from));

        auto proof = index.value().ProofForChunk(i);
        RNET_CHECK_OK(proof);
        if (!VerifyChunk(chunk, proof.value(), index.value().root())) {
            RNET_FAIL("chunk {} of {} failed its own proof", i, index.value().n_chunks());
        }
    }

    // And bytes that are not that chunk must fail the same proof.
    auto extent = index.value().ChunkExtent(0);
    RNET_CHECK_OK(extent);
    std::vector<uint8_t> forged(
        raw.value().begin(),
        raw.value().begin() + static_cast<std::ptrdiff_t>(extent.value().second));
    forged[0] ^= 0xFF;
    auto proof = index.value().ProofForChunk(0);
    RNET_CHECK_OK(proof);
    RNET_CHECK(!VerifyChunk(forged, proof.value(), index.value().root()));
    std::filesystem::remove(path);
}

RNET_TEST(Index, RejectsAnEmptyCorpus) {
    const auto path = std::filesystem::temp_directory_path() / "rnet_test_empty.txt";
    RNET_CHECK_OK(util::WriteFileAtomic(path, std::vector<uint8_t>{}));
    RNET_CHECK_ERR(DatasetIndex::Build(path, 4096));
    std::filesystem::remove(path);
}

// ---------------------------------------------------------------------------
// The manifest, and reading a chunk back.
// ---------------------------------------------------------------------------

RNET_TEST(Manifest, PersistsAndReloads) {
    const auto path = WriteCorpus("persist", 120);
    const auto prefix = std::filesystem::temp_directory_path() / "rnet_test_manifest";
    const auto tokenizer = crypto::Sha3_256(std::string_view("a tokenizer"));

    auto built = BuildAndWriteManifest(path, prefix, 4096, tokenizer);
    RNET_CHECK_OK(built);

    auto loaded = LoadManifestFile(prefix.string() + ".rnds");
    RNET_CHECK_OK(loaded);
    RNET_CHECK_EQ(loaded.value().Id(), built.value().Id());
    RNET_CHECK_EQ(loaded.value().tokenizer_hash, tokenizer);
    RNET_CHECK_EQ(loaded.value().n_bytes, built.value().n_bytes);

    std::filesystem::remove(path);
    std::filesystem::remove(prefix.string() + ".rnds");
    std::filesystem::remove(prefix.string() + ".json");
}

// The bytes a worker is handed must be exactly the chunk: it is those bytes it
// tokenizes and those bytes the root covers. One byte either way and the
// verifier reproduces something else.
RNET_TEST(Manifest, ReadChunkReturnsExactlyTheChunk) {
    const auto path = WriteCorpus("readchunk", 140);
    auto index = DatasetIndex::Build(path, 4096);
    RNET_CHECK_OK(index);
    const auto manifest = index.value().ToManifest(crypto::Hash256{});

    auto raw = util::ReadFile(path);
    RNET_CHECK_OK(raw);

    for (uint64_t i = 0; i < manifest.n_chunks; ++i) {
        auto extent = index.value().ChunkExtent(i);
        RNET_CHECK_OK(extent);
        const uint64_t from = extent.value().first;
        const uint64_t to = extent.value().second;

        auto got = ReadChunk(path, manifest, i);
        RNET_CHECK_OK(got);
        RNET_CHECK_EQ(got.value().size(), static_cast<size_t>(to - from));
        for (size_t j = 0; j < got.value().size(); ++j) {
            if (got.value()[j] != raw.value()[from + j]) {
                RNET_FAIL("chunk {} differs at byte {}", i, j);
            }
        }
    }

    RNET_CHECK_ERR(ReadChunk(path, manifest, manifest.n_chunks));
    std::filesystem::remove(path);
}

// Every chunk the schedule can name must be readable, for every worker and every
// step. A schedule that can point at a chunk nobody can fetch is a worker that
// cannot train and cannot say why.
RNET_TEST(Manifest, ScheduledChunksAreAlwaysReadable) {
    const auto path = WriteCorpus("scheduled", 200);
    auto index = DatasetIndex::Build(path, 4096);
    RNET_CHECK_OK(index);
    const auto manifest = index.value().ToManifest(crypto::Hash256{});

    for (uint64_t worker = 1; worker <= 4; ++worker) {
        for (uint64_t step = 0; step < 8; ++step) {
            auto chunks = BatchChunks(manifest.dataset_root, 0, worker, step, 4, manifest.n_chunks);
            RNET_CHECK_OK(chunks);
            for (uint64_t c : chunks.value()) {
                auto got = ReadChunk(path, manifest, c);
                if (!got) {
                    RNET_FAIL("worker {} step {} was sent to chunk {}: {}", worker, step, c,
                              got.error());
                }
                RNET_CHECK(!got.value().empty());
            }
        }
    }
    std::filesystem::remove(path);
}
