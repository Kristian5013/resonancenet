#include "canon/canon.h"
#include "canon/objects.h"
#include "test/framework.h"
#include "util/hex.h"

using namespace rnet;
using namespace rnet::canon;

RNET_TEST(Canon, IntegersAreBigEndian) {
    Writer w;
    w.U16(0x0102);
    w.U32(0x01020304);
    w.U64(0x0102030405060708ULL);
    RNET_CHECK_STREQ(util::ToHex(w.data()), "0102010203040102030405060708");
}

RNET_TEST(Canon, ReaderRoundTrips) {
    Writer w;
    w.U8(0xAB);
    w.U16(0xBEEF);
    w.U32(0xDEADBEEF);
    w.U64(0x0123456789ABCDEFULL);
    w.String("resonance");

    Reader r(w.data());
    auto u8 = r.U8();
    auto u16 = r.U16();
    auto u32 = r.U32();
    auto u64 = r.U64();
    auto s = r.String();
    RNET_CHECK_OK(u8);
    RNET_CHECK_OK(u16);
    RNET_CHECK_OK(u32);
    RNET_CHECK_OK(u64);
    RNET_CHECK_OK(s);
    RNET_CHECK_EQ(u8.value(), 0xAB);
    RNET_CHECK_EQ(u16.value(), 0xBEEF);
    RNET_CHECK_EQ(u32.value(), 0xDEADBEEFu);
    RNET_CHECK_EQ(u64.value(), 0x0123456789ABCDEFULL);
    RNET_CHECK_STREQ(s.value(), "resonance");
    RNET_CHECK_OK(r.ExpectExhausted());
}

// A reader must never walk off the end — the whole point of bounds-checked reads.
RNET_TEST(Canon, TruncatedInputIsRejected) {
    Writer w;
    w.U32(0x11223344);
    std::vector<uint8_t> bytes = w.take();
    bytes.pop_back();

    Reader r(bytes);
    RNET_CHECK_ERR(r.U32());
}

RNET_TEST(Canon, VarBytesLengthCannotExceedInput) {
    Writer w;
    w.U32(0xFFFFFFF0);   // hostile length prefix
    w.U8(0x01);
    Reader r(w.data());
    RNET_CHECK_ERR(r.VarBytes());
}

RNET_TEST(Canon, ContainerRoundTrips) {
    const std::vector<uint8_t> content{0xDE, 0xAD, 0xBE, 0xEF};
    const auto raw = WrapContainer(ObjectType::DatasetManifest, content);

    auto parsed = ParseContainer(raw);
    RNET_CHECK_OK(parsed);
    RNET_CHECK(parsed.value().type == ObjectType::DatasetManifest);
    RNET_CHECK_EQ(parsed.value().content, content);
    RNET_CHECK_EQ(parsed.value().version, kProtocolVersion);
}

RNET_TEST(Canon, CorruptedByteIsDetected) {
    const std::vector<uint8_t> content{1, 2, 3, 4, 5, 6, 7, 8};
    auto raw = WrapContainer(ObjectType::RoundDescriptor, content);

    // Flip a bit in the payload: the CRC and the content hash must both notice.
    raw[16] ^= 0x01;
    RNET_CHECK_ERR(ParseContainer(raw));
}

RNET_TEST(Canon, TruncatedContainerIsDetected) {
    const std::vector<uint8_t> content{1, 2, 3, 4};
    auto raw = WrapContainer(ObjectType::RoundDescriptor, content);
    raw.resize(raw.size() - 1);
    RNET_CHECK_ERR(ParseContainer(raw));
}

RNET_TEST(Canon, AppendedBytesAreDetected) {
    const std::vector<uint8_t> content{1, 2, 3, 4};
    auto raw = WrapContainer(ObjectType::RoundDescriptor, content);
    raw.push_back(0x00);
    RNET_CHECK_ERR(ParseContainer(raw));
}

RNET_TEST(Canon, BadMagicIsRejected) {
    const std::vector<uint8_t> content{1, 2, 3, 4};
    auto raw = WrapContainer(ObjectType::RoundDescriptor, content);
    raw[0] = 'X';
    RNET_CHECK_ERR(ParseContainer(raw));
}

// CRC32C (Castagnoli) — a project that used the IEEE polynomial by mistake would
// still "work" locally but disagree with every other implementation.
RNET_TEST(Canon, Crc32cKnownAnswer) {
    const std::string_view msg = "123456789";
    const auto span = std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(msg.data()), msg.size());
    RNET_CHECK_EQ(Crc32c(span), 0xE3069283u);
}

RNET_TEST(Objects, RoundDescriptorRoundTrips) {
    RoundDescriptor d;
    d.round_id = 7;
    d.model.d_model = 2048;
    d.model.n_layers = 16;
    d.model.n_heads = 16;
    d.model.n_kv_heads = 4;
    d.model.d_ff = 5632;
    d.model.vocab_size = 128000;
    d.model.seq_len = 8192;
    d.model.rope_theta = 500000;
    d.model.tie_embeddings = 1;
    d.model.qk_norm = 1;

    auto decoded = RoundDescriptor::FromContainer(d.ToContainer());
    RNET_CHECK_OK(decoded);
    RNET_CHECK_EQ(decoded.value().round_id, 7u);
    RNET_CHECK_EQ(decoded.value().model.d_model, 2048u);
    RNET_CHECK_EQ(decoded.value().model.vocab_size, 128000u);
    RNET_CHECK_EQ(decoded.value().Id(), d.Id());
}

RNET_TEST(Objects, InvalidModelIsRejectedOnDecode) {
    RoundDescriptor d;
    d.round_id = 1;
    d.model.d_model = 2048;
    d.model.n_layers = 16;
    d.model.n_heads = 15;      // does not divide d_model
    d.model.n_kv_heads = 5;
    d.model.d_ff = 5632;
    d.model.vocab_size = 128000;
    d.model.seq_len = 8192;
    d.model.rope_theta = 500000;

    RNET_CHECK_ERR(RoundDescriptor::FromContainer(d.ToContainer()));
}

RNET_TEST(Objects, ContainerTypeIsEnforced) {
    DatasetManifest m;
    m.n_tokens = 4096;
    m.chunk_tokens = 1024;
    m.n_chunks = 4;
    RNET_CHECK_ERR(RoundDescriptor::FromContainer(m.ToContainer()));
}

RNET_TEST(Objects, DatasetManifestRejectsInconsistentChunkCount) {
    DatasetManifest m;
    m.n_tokens = 4096;
    m.chunk_tokens = 1024;
    m.n_chunks = 3;            // should be 4
    RNET_CHECK_ERR(DatasetManifest::FromContainer(m.ToContainer()));
}

// Any change to a field must change the object's identity, or two different
// configurations could masquerade as the same round.
RNET_TEST(Objects, EveryFieldAffectsIdentity) {
    RoundDescriptor base;
    base.model.d_model = 2048;
    base.model.n_layers = 16;
    base.model.n_heads = 16;
    base.model.n_kv_heads = 4;
    base.model.d_ff = 5632;
    base.model.vocab_size = 128000;
    base.model.seq_len = 8192;
    base.model.rope_theta = 500000;
    base.model.tie_embeddings = 1;
    base.model.qk_norm = 1;
    const auto id = base.Id();

    auto changed = base;
    changed.model.qk_norm = 0;
    RNET_CHECK(changed.Id() != id);

    changed = base;
    changed.model.rope_theta = 10000;
    RNET_CHECK(changed.Id() != id);

    changed = base;
    changed.tokenizer_hash[0] = 0x01;
    RNET_CHECK(changed.Id() != id);

    changed = base;
    changed.dataset_root[31] = 0x01;
    RNET_CHECK(changed.Id() != id);
}

RNET_TEST(Objects, ParameterCountMatchesHandCalculation) {
    ModelSpec m;
    m.d_model = 2048;
    m.n_layers = 16;
    m.n_heads = 16;
    m.n_kv_heads = 4;
    m.d_ff = 5632;
    m.vocab_size = 128000;
    m.seq_len = 8192;
    m.rope_theta = 500000;
    m.tie_embeddings = 1;
    m.qk_norm = 1;
    RNET_CHECK_OK(m.Validate());

    // ~1.0B for RN-1B: embeddings 262M + 16 layers x ~46M.
    const uint64_t params = m.ParameterCount();
    RNET_CHECK(params > 900'000'000ull);
    RNET_CHECK(params < 1'100'000'000ull);
}
