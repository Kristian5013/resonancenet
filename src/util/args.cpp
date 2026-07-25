#include "util/args.h"

#include <charconv>
#include <format>

namespace rnet::util {

void ArgParser::AddOption(std::string name, std::string help, bool takes_value,
                          std::string default_value) {
    declared_[name] = Option{std::move(help), takes_value, default_value};
    if (!default_value.empty()) values_[std::move(name)] = std::move(default_value);
}

void ArgParser::AddRepeated(std::string name, std::string help) {
    declared_[std::move(name)] = Option{std::move(help), /*takes_value=*/true, {}, /*repeated=*/true};
}

const std::vector<std::string>& ArgParser::GetRepeated(std::string_view name) const {
    static const std::vector<std::string> kEmpty;
    const auto it = repeated_.find(std::string(name));
    return it == repeated_.end() ? kEmpty : it->second;
}

void ArgParser::AddPositional(std::string name, std::string help) {
    positional_spec_.emplace_back(std::move(name), std::move(help));
}

Status ArgParser::Parse(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (!arg.starts_with("--")) {
            positionals_.push_back(std::move(arg));
            continue;
        }

        std::string name = arg.substr(2);
        std::string inline_value;
        bool has_inline = false;
        if (const auto eq = name.find('='); eq != std::string::npos) {
            inline_value = name.substr(eq + 1);
            name = name.substr(0, eq);
            has_inline = true;
        }

        const auto decl = declared_.find(name);
        if (decl == declared_.end()) {
            return Err("unknown option --" + name + "\n\n" + Usage());
        }

        if (!decl->second.takes_value) {
            if (has_inline) return Err("--" + name + " does not take a value");
            values_[name] = "1";
            continue;
        }

        if (has_inline) {
            if (decl->second.repeated) {
                repeated_[name].push_back(inline_value);
            }
            values_[name] = std::move(inline_value);
            continue;
        }
        if (i + 1 >= argc) return Err("--" + name + " requires a value");
        if (decl->second.repeated) repeated_[name].emplace_back(argv[i + 1]);
        values_[name] = argv[++i];
    }
    return Status::Ok();
}

bool ArgParser::Has(std::string_view name) const {
    return values_.find(std::string(name)) != values_.end();
}

bool ArgParser::GetBool(std::string_view name) const {
    const auto it = values_.find(std::string(name));
    return it != values_.end() && it->second == "1";
}

std::string ArgParser::GetString(std::string_view name, std::string fallback) const {
    const auto it = values_.find(std::string(name));
    return it == values_.end() ? std::move(fallback) : it->second;
}

Result<int64_t> ArgParser::GetInt(std::string_view name) const {
    const auto it = values_.find(std::string(name));
    if (it == values_.end()) return Err("missing --" + std::string(name));
    int64_t v = 0;
    const auto& s = it->second;
    const auto res = std::from_chars(s.data(), s.data() + s.size(), v);
    if (res.ec != std::errc() || res.ptr != s.data() + s.size()) {
        return Err("--" + std::string(name) + ": not an integer: " + s);
    }
    return v;
}

Result<uint64_t> ArgParser::GetUInt(std::string_view name) const {
    auto v = GetInt(name);
    if (!v) return Err(v.error());
    if (v.value() < 0) return Err("--" + std::string(name) + ": must be non-negative");
    return static_cast<uint64_t>(v.value());
}

std::string ArgParser::Usage() const {
    std::string out = std::format("{}\n\nUsage: {}", summary_, program_);
    for (const auto& [name, help] : positional_spec_) out += std::format(" <{}>", name);
    out += " [options]\n";
    if (!positional_spec_.empty()) {
        out += "\nArguments:\n";
        for (const auto& [name, help] : positional_spec_) {
            out += std::format("  {:<22} {}\n", name, help);
        }
    }
    out += "\nOptions:\n";
    for (const auto& [name, opt] : declared_) {
        std::string flag = opt.takes_value ? std::format("--{} <value>", name) : std::format("--{}", name);
        out += std::format("  {:<22} {}", flag, opt.help);
        if (!opt.default_value.empty()) out += std::format(" (default: {})", opt.default_value);
        out += "\n";
    }
    return out;
}

}  // namespace rnet::util
