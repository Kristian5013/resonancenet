#include <cstdio>
#include <string>

#include "test/framework.h"

namespace rnet::test {

std::vector<TestCase>& Registry() {
    static std::vector<TestCase> registry;
    return registry;
}

int RunAll(const std::string& filter) {
    int passed = 0;
    int failed = 0;
    std::string current_suite;

    for (const auto& tc : Registry()) {
        const std::string full = tc.suite + "." + tc.name;
        if (!filter.empty() && full.find(filter) == std::string::npos) continue;

        if (tc.suite != current_suite) {
            current_suite = tc.suite;
            std::printf("[%s]\n", current_suite.c_str());
        }
        try {
            tc.body();
            std::printf("  ok    %s\n", tc.name.c_str());
            ++passed;
        } catch (const AssertionFailure& f) {
            std::printf("  FAIL  %s\n        %s\n", tc.name.c_str(), f.message.c_str());
            ++failed;
        } catch (const std::exception& e) {
            std::printf("  FAIL  %s\n        unexpected exception: %s\n", tc.name.c_str(), e.what());
            ++failed;
        }
    }

    std::printf("\n%d passed, %d failed\n", passed, failed);
    return failed == 0 ? 0 : 1;
}

}  // namespace rnet::test

int main(int argc, char** argv) {
    const std::string filter = argc > 1 ? argv[1] : "";
    return rnet::test::RunAll(filter);
}
