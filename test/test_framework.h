#pragma once

// Minimal test framework for ResonanceNet — no external dependencies.
// Usage:
//   TEST(my_test) { ASSERT_TRUE(1 + 1 == 2); }
//   int main() { return rnet::test::run_all_tests(); }

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace rnet::test {

// ─── Test registry ──────────────────────────────────────────────────

struct TestEntry {
    std::string name;
    std::function<void()> func;
};

inline std::vector<TestEntry>& test_registry() {
    static std::vector<TestEntry> tests;
    return tests;
}

struct TestRegistrar {
    TestRegistrar(const char* name, std::function<void()> fn) {
        test_registry().push_back({name, std::move(fn)});
    }
};

// ─── Test runner ────────────────────────────────────────────────────

inline int run_all_tests() {
    auto& tests = test_registry();
    int passed = 0;
    int failed = 0;
    int total = static_cast<int>(tests.size());

    printf("Running %d tests...\n\n", total);

    for (auto& t : tests) {
        try {
            t.func();
            printf("  PASS: %s\n", t.name.c_str());
            passed++;
        } catch (const std::exception& e) {
            printf("  FAIL: %s\n        %s\n", t.name.c_str(), e.what());
            failed++;
        } catch (...) {
            printf("  FAIL: %s\n        (unknown exception)\n", t.name.c_str());
            failed++;
        }
    }

    printf("\n%d/%d tests passed", passed, total);
    if (failed > 0) {
        printf(", %d FAILED", failed);
    }
    printf(".\n");

    return failed > 0 ? 1 : 0;
}

// ─── Assertion exception ────────────────────────────────────────────

class AssertionFailure : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

}  // namespace rnet::test

// ─── Macros ─────────────────────────────────────────────────────────

#define TEST(name) \
    static void test_func_##name(); \
    static rnet::test::TestRegistrar test_reg_##name(#name, test_func_##name); \
    static void test_func_##name()

#define ASSERT_TRUE(expr) \
    do { \
        if (!(expr)) { \
            throw rnet::test::AssertionFailure( \
                std::string(__FILE__) + ":" + std::to_string(__LINE__) + \
                ": ASSERT_TRUE(" #expr ") failed"); \
        } \
    } while (0)

#define ASSERT_FALSE(expr) \
    do { \
        if ((expr)) { \
            throw rnet::test::AssertionFailure( \
                std::string(__FILE__) + ":" + std::to_string(__LINE__) + \
                ": ASSERT_FALSE(" #expr ") failed"); \
        } \
    } while (0)

#define ASSERT_EQ(a, b) \
    do { \
        auto&& _a = (a); \
        auto&& _b = (b); \
        if (!(_a == _b)) { \
            throw rnet::test::AssertionFailure( \
                std::string(__FILE__) + ":" + std::to_string(__LINE__) + \
                ": ASSERT_EQ(" #a ", " #b ") failed"); \
        } \
    } while (0)

#define ASSERT_NE(a, b) \
    do { \
        auto&& _a = (a); \
        auto&& _b = (b); \
        if (_a == _b) { \
            throw rnet::test::AssertionFailure( \
                std::string(__FILE__) + ":" + std::to_string(__LINE__) + \
                ": ASSERT_NE(" #a ", " #b ") failed"); \
        } \
    } while (0)

#define ASSERT_NEAR(a, b, epsilon) \
    do { \
        auto _a = static_cast<double>(a); \
        auto _b = static_cast<double>(b); \
        if (std::fabs(_a - _b) > static_cast<double>(epsilon)) { \
            throw rnet::test::AssertionFailure( \
                std::string(__FILE__) + ":" + std::to_string(__LINE__) + \
                ": ASSERT_NEAR(" #a ", " #b ", " #epsilon ") failed: |" + \
                std::to_string(_a) + " - " + std::to_string(_b) + "| > " + \
                std::to_string(static_cast<double>(epsilon))); \
        } \
    } while (0)

#define ASSERT_THROWS(expr) \
    do { \
        bool _threw = false; \
        try { expr; } catch (...) { _threw = true; } \
        if (!_threw) { \
            throw rnet::test::AssertionFailure( \
                std::string(__FILE__) + ":" + std::to_string(__LINE__) + \
                ": ASSERT_THROWS(" #expr ") failed — no exception thrown"); \
        } \
    } while (0)
