#include <cstdio>
#include <filesystem>
#include <random>

#include "dataset/index.h"
#include "dataset/manifest.h"
#include "dataset/scheduler.h"
#include "test/framework.h"
#include "util/fs.h"
#include "util/hex.h"

using namespace rnet;
using namespace rnet::dataset;

namespace {

// A scratch corpus of `n` little-endian uint32 tokens, deterministic in content.
std::filesystem::path WriteCorpus(const std::string& name, uint64_t n) {
    const auto path = std::filesystem::temp_directory_path() / ("rnet_test_" + name + ".bin");
    std::vector<uint8_t> bytes(static_cast<size_t>(n) * 4);
    std::mt19937 rng(1234);
    for (uint64_t i = 0; i < n; ++i) {
        const uint32_t tok = rng() % 128000u;
        bytes[i * 4 + 0] = static_cast<uint8_t>(tok);
        bytes[i * 4 + 1] = static_cast<uint8_t>(tok >> 8);
        bytes[i * 4 + 2] = static_cast<uint8_t>(tok >> 16);
        bytes[i * 4 + 3] = static_cast<uint8_t>(tok >> 24);
    }
    (void)util::WriteFileAtomic(path, bytes);
    return path;
}

const crypto::Hash256 kRoot = crypto::Sha3_256(std::string_view("test-dataset-root"));

}  // namespace

// ---------------------------------------------------------------------------
// Deterministic batch derivation — the property that closes data poisoning.
// ---------------------------------------------------------------------------
RNET_TEST(Schedule, OffsetsAreDeterministic) {
    auto a = BatchOffsets(kRoot, 0, 7, 42, 4, 1'000'000, 8192);
    auto b = BatchOffsets(kRoot, 0, 7, 42, 4, 1'000'000, 8192);
    RNET_CHECK_OK(a);
    RNET_CHECK_OK(b);
    RNET_CHECK_EQ(a.value(), b.value());
}

RNET_TEST(Schedule, DifferentWorkersGetDifferentData) {
    auto w1 = BatchOffsets(kRoot, 0, 1, 5, 8, 1'000'000, 1024);
    auto w2 = BatchOffsets(kRoot, 0, 2, 5, 8, 1'000'000, 1024);
    RNET_CHECK_OK(w1);
    RNET_CHECK_OK(w2);
    RNET_CHECK(w1.value() != w2.value());
}

RNET_TEST(Schedule, EveryInputAffectsTheOffset) {
    const uint64_t n = 10'000'000;
    const uint32_t seq = 1024;
    auto base = WindowOffset(kRoot, 3, 9, 100, 0, n, seq);
    RNET_CHECK_OK(base);

    crypto::Hash256 other_root = kRoot;
    other_root[0] ^= 0x01;
    for (auto v : {WindowOffset(other_root, 3, 9, 100, 0, n, seq),
                   WindowOffset(kRoot, 4, 9, 100, 0, n, seq),
                   WindowOffset(kRoot, 3, 10, 100, 0, n, seq),
                   WindowOffset(kRoot, 3, 9, 101, 0, n, seq),
                   WindowOffset(kRoot, 3, 9, 100, 1, n, seq)}) {
        RNET_CHECK_OK(v);
        RNET_CHECK(v.value() != base.value());
    }
}

RNET_TEST(Schedule, OffsetsStayInBounds) {
    const uint64_t n = 50'000;
    const uint32_t seq = 512;
    auto windows = WindowCount(n, seq);
    RNET_CHECK_OK(windows);
    for (uint64_t step = 0; step < 200; ++step) {
        auto offs = BatchOffsets(kRoot, 0, 1, step, 8, n, seq);
        RNET_CHECK_OK(offs);
        for (uint64_t off : offs.value()) {
            RNET_CHECK(off < windows.value());
            RNET_CHECK(off + seq + 1 <= n);   // window always fits
        }
    }
}

RNET_TEST(Schedule, CorpusTooSmallIsAnError) {
    RNET_CHECK_ERR(WindowCount(100, 8192));
    RNET_CHECK_ERR(WindowCount(8193, 8192));
    RNET_CHECK_ERR(BatchOffsets(kRoot, 0, 0, 0, 1, 100, 8192));
    RNET_CHECK_ERR(BatchOffsets(kRoot, 0, 0, 0, 0, 100'000, 512));   // micro_batch = 0
}

// Cross-language anchor: these exact offsets must also be produced by the Python
// worker. If this vector changes, every node must be upgraded in lockstep.
RNET_TEST(Schedule, GoldenVector) {
    const crypto::Hash256 root = crypto::Sha3_256(std::string_view("resonancenet-golden"));
    auto offs = BatchOffsets(root, 1, 2, 3, 4, 1'000'000, 512);
    RNET_CHECK_OK(offs);
    RNET_CHECK_EQ(offs.value().size(), size_t{4});
    // Recorded from this implementation; the Python mirror is asserted against it.
    std::printf("        [golden offsets] %llu %llu %llu %llu\n",
                static_cast<unsigned long long>(offs.value()[0]),
                static_cast<unsigned long long>(offs.value()[1]),
                static_cast<unsigned long long>(offs.value()[2]),
                static_cast<unsigned long long>(offs.value()[3]));
}

// ---------------------------------------------------------------------------
// Corpus integrity
// ---------------------------------------------------------------------------
RNET_TEST(Index, BuildsAndVerifies) {
    const auto path = WriteCorpus("index", 40'000);
    auto index = DatasetIndex::Build(path, 4096, canon::TokenDtype::Uint32);
    RNET_CHECK_OK(index);
    RNET_CHECK_EQ(index.value().n_tokens(), 40'000ull);
    RNET_CHECK_EQ(index.value().n_chunks(), 10u);   // ceil(40000/4096) = 10

    const auto manifest = index.value().ToManifest(crypto::Hash256{});
    RNET_CHECK_OK(manifest.Validate());
    RNET_CHECK_OK(VerifyCorpus(path, manifest));
    std::filesystem::remove(path);
}

RNET_TEST(Index, AlteredCorpusFailsVerification) {
    const auto path = WriteCorpus("altered", 20'000);
    auto index = DatasetIndex::Build(path, 4096, canon::TokenDtype::Uint32);
    RNET_CHECK_OK(index);
    const auto manifest = index.value().ToManifest(crypto::Hash256{});

    auto bytes = util::ReadFile(path);
    RNET_CHECK_OK(bytes);
    auto data = bytes.value();
    data[12345] ^= 0x01;                       // flip one bit deep inside the corpus
    RNET_CHECK_OK(util::WriteFileAtomic(path, data));

    // A seed node that serves altered data must be caught by the root.
    auto st = VerifyCorpus(path, manifest);
    RNET_CHECK(!st);
    std::filesystem::remove(path);
}

RNET_TEST(Index, ChunkProofsVerifyAgainstRoot) {
    const auto path = WriteCorpus("proofs", 30'000);
    auto index = DatasetIndex::Build(path, 4096, canon::TokenDtype::Uint32);
    RNET_CHECK_OK(index);
    const auto& idx = index.value();

    for (uint64_t chunk = 0; chunk < idx.n_chunks(); ++chunk) {
        auto proof = idx.ProofForChunk(chunk);
        RNET_CHECK_OK(proof);

        const uint64_t offset_bytes = chunk * idx.chunk_tokens() * 4ull;
        const uint64_t remaining_tokens = idx.n_tokens() - chunk * idx.chunk_tokens();
        const size_t want = static_cast<size_t>(
            std::min<uint64_t>(remaining_tokens, idx.chunk_tokens()) * 4ull);
        auto chunk_bytes = util::ReadFileRange(path, offset_bytes, want);
        RNET_CHECK_OK(chunk_bytes);

        if (!VerifyChunk(chunk_bytes.value(), proof.value(), idx.root())) {
            RNET_FAIL("chunk {} failed to verify against dataset root", chunk);
        }
    }
    std::filesystem::remove(path);
}

RNET_TEST(Index, RejectsTruncatedTokenFile) {
    const auto path = std::filesystem::temp_directory_path() / "rnet_test_ragged.bin";
    const std::vector<uint8_t> ragged(4 * 10 + 2, 0x01);   // not a whole number of uint32
    RNET_CHECK_OK(util::WriteFileAtomic(path, ragged));
    RNET_CHECK_ERR(DatasetIndex::Build(path, 4096, canon::TokenDtype::Uint32));
    std::filesystem::remove(path);
}

RNET_TEST(Manifest, PersistsAndReloads) {
    const auto path = WriteCorpus("manifest", 12'000);
    const auto prefix = std::filesystem::temp_directory_path() / "rnet_test_manifest";
    const crypto::Hash256 tok = crypto::Sha3_256(std::string_view("tokenizer"));

    auto built = BuildAndWriteManifest(path, prefix, 4096, canon::TokenDtype::Uint32, tok);
    RNET_CHECK_OK(built);

    auto loaded = LoadManifestFile(prefix.string() + ".rnds");
    RNET_CHECK_OK(loaded);
    RNET_CHECK_EQ(loaded.value().dataset_root, built.value().dataset_root);
    RNET_CHECK_EQ(loaded.value().tokenizer_hash, tok);
    RNET_CHECK_EQ(loaded.value().n_tokens, 12'000ull);

    std::filesystem::remove(path);
    std::filesystem::remove(prefix.string() + ".rnds");
    std::filesystem::remove(prefix.string() + ".json");
}

RNET_TEST(Manifest, ReadWindowReturnsExactTokens) {
    const auto path = WriteCorpus("window", 5'000);
    auto index = DatasetIndex::Build(path, 1024, canon::TokenDtype::Uint32);
    RNET_CHECK_OK(index);
    const auto manifest = index.value().ToManifest(crypto::Hash256{});

    auto window = ReadWindow(path, manifest, 100, 256);
    RNET_CHECK_OK(window);
    RNET_CHECK_EQ(window.value().size(), size_t{257});   // seq_len + shifted target

    // Reading past the end must fail rather than return short or zero-padded data.
    RNET_CHECK_ERR(ReadWindow(path, manifest, 4'900, 256));
    std::filesystem::remove(path);
}

RNET_TEST(Manifest, ScheduledWindowsAreAlwaysReadable) {
    const auto path = WriteCorpus("scheduled", 20'000);
    auto index = DatasetIndex::Build(path, 4096, canon::TokenDtype::Uint32);
    RNET_CHECK_OK(index);
    const auto manifest = index.value().ToManifest(crypto::Hash256{});
    const uint32_t seq = 512;

    // End to end: the schedule picks offsets, and every one of them is a valid,
    // fully-readable window — no worker choice anywhere in the path.
    for (uint64_t step = 0; step < 50; ++step) {
        auto offs = BatchOffsets(manifest.dataset_root, 0, 3, step, 2, manifest.n_tokens, seq);
        RNET_CHECK_OK(offs);
        for (uint64_t off : offs.value()) {
            auto w = ReadWindow(path, manifest, off, seq);
            RNET_CHECK_OK(w);
            RNET_CHECK_EQ(w.value().size(), static_cast<size_t>(seq) + 1);
        }
    }
    std::filesystem::remove(path);
}
