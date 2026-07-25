// Minimal self-registering test framework.
//
// Dependency-free on purpose: the test binary must build anywhere the project
// builds, with no package manager involved.
#pragma once

#include <format>
#include <functional>
#include <string>
#include <vector>

namespace rnet::test {

struct TestCase {
    std::string suite;
    std::string name;
    std::function<void()> body;
};

// Thrown by the assertion macros; caught by the runner, which reports the case as
// failed and continues with the next one.
struct AssertionFailure {
    std::string message;
};

std::vector<TestCase>& Registry();

struct Registrar {
    Registrar(std::string suite, std::string name, std::function<void()> body) {
        Registry().push_back({std::move(suite), std::move(name), std::move(body)});
    }
};

int RunAll(const std::string& filter);

#define RNET_TEST(suite, name)                                                      \
    static void suite##_##name##_body();                                            \
    static const ::rnet::test::Registrar suite##_##name##_registrar(                \
        #suite, #name, suite##_##name##_body);                                      \
    static void suite##_##name##_body()

#define RNET_FAIL(...) \
    throw ::rnet::test::AssertionFailure{std::format("{}:{}: {}", __FILE__, __LINE__, std::format(__VA_ARGS__))}

#define RNET_CHECK(cond)                                          \
    do {                                                          \
        if (!(cond)) RNET_FAIL("check failed: {}", #cond);        \
    } while (0)

#define RNET_CHECK_EQ(a, b)                                                             \
    do {                                                                                \
        const auto& lhs_ = (a);                                                         \
        const auto& rhs_ = (b);                                                         \
        if (!(lhs_ == rhs_)) RNET_FAIL("{} != {}", #a, #b);                             \
    } while (0)

#define RNET_CHECK_STREQ(a, b)                                                          \
    do {                                                                                \
        const std::string lhs_ = (a);                                                   \
        const std::string rhs_ = (b);                                                    \
        if (lhs_ != rhs_) RNET_FAIL("{}\n  actual:   {}\n  expected: {}", #a, lhs_, rhs_); \
    } while (0)

// The result must be ok(); on failure the reason is reported.
// Binds by forwarding reference: the checked expression may return a move-only
// type (a Result holding a socket, say), which a by-value capture could not take.
#define RNET_CHECK_OK(...)                                                              \
    do {                                                                                \
        auto&& res_ = (__VA_ARGS__);                                                    \
        if (!res_) RNET_FAIL("{} failed: {}", #__VA_ARGS__, res_.error());              \
    } while (0)

// The result must FAIL — used to prove that malformed input is rejected rather
// than silently accepted.
#define RNET_CHECK_ERR(...)                                                             \
    do {                                                                                \
        auto&& res_ = (__VA_ARGS__);                                                    \
        if (res_) RNET_FAIL("{} unexpectedly succeeded", #__VA_ARGS__);                 \
    } while (0)

}  // namespace rnet::test
