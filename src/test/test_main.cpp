#include <algorithm>
#include <cstdio>
#include <string>

#include "test/framework.h"

namespace rnet::test {

std::vector<TestCase>& Registry() {
    static std::vector<TestCase> registry;
    return registry;
}

// Every test in the binary, one "Suite.Name" per line, sorted.
//
// The count alone is a poor canary. It went from 300 to 297 once, when agents
// reverting their own probe tests restored a whole file and took three tests a
// human had written with it; the fixes those tests covered stayed in the code and
// the proof of them vanished. It was noticed by luck. A count says something died
// but not what, and only if someone happens to be comparing.
//
// A sorted list, checked in and diffed by CI, names the casualty.
void ListAll() {
    std::vector<std::string> names;
    names.reserve(Registry().size());
    for (const auto& tc : Registry()) names.push_back(tc.suite + "." + tc.name);
    std::sort(names.begin(), names.end());
    for (const auto& name : names) std::printf("%s\n", name.c_str());
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

    // A build that omits the transport half still prints a green count, and that
    // number is a lie by omission: sockets are not available on every platform, so
    // the net, protocol and IPC suites are compiled in conditionally, and a report
    // of "all passed" from a binary that never contained them reads exactly like a
    // report from one that did. Said out loud rather than left to be inferred from
    // a smaller number nobody has the other number to compare against.
#ifdef RNET_TESTS_TRANSPORT
    std::printf("suite: consensus + transport (complete)\n");
#else
    std::printf(
        "suite: consensus ONLY — the net, protocol and IPC tests were not compiled into this "
        "binary, so this result says nothing about them\n");
#endif
    return failed == 0 ? 0 : 1;
}

}  // namespace rnet::test

int main(int argc, char** argv) {
    const std::string arg = argc > 1 ? argv[1] : "";
    if (arg == "--list") {
        rnet::test::ListAll();
        return 0;
    }
    return rnet::test::RunAll(arg);
}
