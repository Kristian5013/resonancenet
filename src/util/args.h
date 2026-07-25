// Command-line parsing where an unknown flag is an ERROR.
//
// The Lattica audit found a parser that silently ignored unrecognised options, so
// "--warmpu 200" (typo) looked like it worked and changed nothing. Here every
// option must be declared, and anything unrecognised aborts with usage.
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "util/result.h"

namespace rnet::util {

class ArgParser {
public:
    ArgParser(std::string program, std::string summary)
        : program_(std::move(program)), summary_(std::move(summary)) {}

    // `takes_value == false` declares a boolean flag ("--fast").
    void AddOption(std::string name, std::string help, bool takes_value = true,
                   std::string default_value = {});
    // An option that may appear more than once and keeps every value, rather than
    // the last silently winning. `--connect a --connect b` must mean both, not b.
    void AddRepeated(std::string name, std::string help);
    void AddPositional(std::string name, std::string help);

    Status Parse(int argc, char** argv);

    bool Has(std::string_view name) const;
    bool GetBool(std::string_view name) const;
    std::string GetString(std::string_view name, std::string fallback = {}) const;
    Result<int64_t> GetInt(std::string_view name) const;
    Result<uint64_t> GetUInt(std::string_view name) const;

    // Every value given for a repeated option, in the order it appeared.
    const std::vector<std::string>& GetRepeated(std::string_view name) const;

    const std::vector<std::string>& positionals() const { return positionals_; }
    std::string Usage() const;

private:
    struct Option {
        std::string help;
        bool takes_value{true};
        std::string default_value;
        bool repeated{false};
    };

    std::string program_;
    std::string summary_;
    std::map<std::string, Option> declared_;
    std::vector<std::pair<std::string, std::string>> positional_spec_;
    std::map<std::string, std::string> values_;
    std::map<std::string, std::vector<std::string>> repeated_;
    std::vector<std::string> positionals_;
};

}  // namespace rnet::util
