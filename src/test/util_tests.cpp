#include <filesystem>

#include "test/framework.h"
#include "util/args.h"
#include "util/fs.h"
#include "util/hex.h"
#include "util/json.h"

using namespace rnet;
using namespace rnet::util;

RNET_TEST(Hex, RoundTrips) {
    const std::vector<uint8_t> bytes{0x00, 0x0F, 0xF0, 0xFF, 0xAB};
    RNET_CHECK_STREQ(ToHex(bytes), "000ff0ffab");
    auto back = FromHex("000ff0ffab");
    RNET_CHECK_OK(back);
    RNET_CHECK_EQ(back.value(), bytes);
}

RNET_TEST(Hex, MalformedInputIsRejected) {
    RNET_CHECK_ERR(FromHex("abc"));       // odd length
    RNET_CHECK_ERR(FromHex("zz"));        // not hex
    RNET_CHECK_ERR(FromHexFixed<32>("aabb"));   // wrong width
}

RNET_TEST(Json, WritesAndParses) {
    Json obj = Json::Object();
    obj.Set("name", "resonance");
    obj.Set("count", int64_t{42});
    obj.Set("ok", true);
    Json arr = Json::Array();
    arr.Push(int64_t{1});
    arr.Push(int64_t{2});
    obj.Set("items", arr);

    auto parsed = Json::Parse(obj.Dump());
    RNET_CHECK_OK(parsed);
    auto name = parsed.value().At("name");
    RNET_CHECK_OK(name);
    RNET_CHECK_STREQ(name.value().AsString().value(), "resonance");
    auto count = parsed.value().At("count");
    RNET_CHECK_OK(count);
    RNET_CHECK_EQ(count.value().AsInt().value(), int64_t{42});
}

// Large integers must survive exactly — routing them through a double would
// corrupt token counts and, later, balances.
RNET_TEST(Json, LargeIntegersKeepPrecision) {
    const int64_t big = 9'007'199'254'740'993ll;   // 2^53 + 1
    Json obj = Json::Object();
    obj.Set("n", big);
    auto parsed = Json::Parse(obj.Dump());
    RNET_CHECK_OK(parsed);
    RNET_CHECK_EQ(parsed.value().At("n").value().AsInt().value(), big);
}

RNET_TEST(Json, RejectsMalformedInput) {
    RNET_CHECK_ERR(Json::Parse("{"));
    RNET_CHECK_ERR(Json::Parse("{\"a\":1"));
    RNET_CHECK_ERR(Json::Parse("{\"a\":1} trailing"));
    RNET_CHECK_ERR(Json::Parse("\"unterminated"));
    RNET_CHECK_ERR(Json::Parse("[1,2,]"));
    RNET_CHECK_ERR(Json::Parse("nul"));
}

RNET_TEST(Json, EscapesRoundTrip) {
    Json obj = Json::Object();
    obj.Set("s", std::string("line\nquote\"tab\tbackslash\\"));
    auto parsed = Json::Parse(obj.Dump());
    RNET_CHECK_OK(parsed);
    RNET_CHECK_STREQ(parsed.value().At("s").value().AsString().value(),
                     "line\nquote\"tab\tbackslash\\");
}

RNET_TEST(Json, WrongTypeIsAnError) {
    auto parsed = Json::Parse("{\"a\": \"text\"}");
    RNET_CHECK_OK(parsed);
    RNET_CHECK_ERR(parsed.value().At("a").value().AsInt());
    RNET_CHECK_ERR(parsed.value().At("missing"));
}

RNET_TEST(Fs, AtomicWriteAndRead) {
    const auto path = std::filesystem::temp_directory_path() / "rnet_test_fs.bin";
    const std::vector<uint8_t> data{1, 2, 3, 4, 5};
    RNET_CHECK_OK(WriteFileAtomic(path, data));

    auto read = ReadFile(path);
    RNET_CHECK_OK(read);
    RNET_CHECK_EQ(read.value(), data);

    auto range = ReadFileRange(path, 1, 3);
    RNET_CHECK_OK(range);
    RNET_CHECK_EQ(range.value(), (std::vector<uint8_t>{2, 3, 4}));

    // A short read must be an error, never silently padded — this is the exact
    // defect pattern found in the Lattica checkpoint loader.
    RNET_CHECK_ERR(ReadFileRange(path, 3, 10));

    std::filesystem::remove(path);
}

RNET_TEST(Fs, MissingFileIsAnError) {
    RNET_CHECK_ERR(ReadFile("/nonexistent/rnet/definitely/not/here.bin"));
    RNET_CHECK_ERR(FileSize("/nonexistent/rnet/definitely/not/here.bin"));
}

RNET_TEST(Args, ParsesOptionsAndPositionals) {
    ArgParser p("rnet-tool", "test");
    p.AddOption("network", "network");
    p.AddOption("fast", "flag", false);
    p.AddOption("count", "number");

    const char* argv[] = {"rnet-tool", "emit", "--network", "regtest", "--fast", "--count=7"};
    RNET_CHECK_OK(p.Parse(6, const_cast<char**>(argv)));

    RNET_CHECK_STREQ(p.GetString("network"), "regtest");
    RNET_CHECK(p.GetBool("fast"));
    RNET_CHECK_EQ(p.GetUInt("count").value(), 7ull);
    RNET_CHECK_EQ(p.positionals().size(), size_t{1});
    RNET_CHECK_STREQ(p.positionals()[0], "emit");
}

// A typo in a flag must abort, not be silently ignored — the Lattica audit found
// exactly this hole ("--warmup" parsed, then thrown away).
RNET_TEST(Args, UnknownOptionIsAnError) {
    ArgParser p("rnet-tool", "test");
    p.AddOption("network", "network");

    const char* argv[] = {"rnet-tool", "--netwrok", "regtest"};
    RNET_CHECK_ERR(p.Parse(3, const_cast<char**>(argv)));
}

RNET_TEST(Args, MissingValueIsAnError) {
    ArgParser p("rnet-tool", "test");
    p.AddOption("network", "network");
    const char* argv[] = {"rnet-tool", "--network"};
    RNET_CHECK_ERR(p.Parse(2, const_cast<char**>(argv)));
}

RNET_TEST(Args, NonNumericIntIsAnError) {
    ArgParser p("rnet-tool", "test");
    p.AddOption("count", "number");
    const char* argv[] = {"rnet-tool", "--count", "12x"};
    RNET_CHECK_OK(p.Parse(3, const_cast<char**>(argv)));
    RNET_CHECK_ERR(p.GetInt("count"));
}
