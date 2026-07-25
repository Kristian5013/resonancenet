#include "util/fs.h"

#include <cerrno>
#include <cstdio>
#include <cstring>

namespace rnet::util {

namespace fs = std::filesystem;

namespace {

std::string Errno() { return std::strerror(errno); }

class File {
public:
    File(const fs::path& path, const char* mode) : f_(std::fopen(path.c_str(), mode)) {}
    ~File() { if (f_) std::fclose(f_); }
    File(const File&) = delete;
    File& operator=(const File&) = delete;

    std::FILE* get() const { return f_; }
    explicit operator bool() const { return f_ != nullptr; }

    // Closes explicitly so the caller can observe a flush/close failure — a
    // deferred close in the destructor would swallow ENOSPC.
    bool CloseChecked() {
        if (!f_) return true;
        const bool ok = std::fclose(f_) == 0;
        f_ = nullptr;
        return ok;
    }

private:
    std::FILE* f_{nullptr};
};

}  // namespace

Result<uint64_t> FileSize(const fs::path& path) {
    std::error_code ec;
    const auto size = fs::file_size(path, ec);
    if (ec) return Err("stat " + path.string() + ": " + ec.message());
    return static_cast<uint64_t>(size);
}

Result<std::vector<uint8_t>> ReadFile(const fs::path& path) {
    auto size = FileSize(path);
    if (!size) return Err(size.error());

    File f(path, "rb");
    if (!f) return Err("open " + path.string() + ": " + Errno());

    std::vector<uint8_t> data(static_cast<size_t>(size.value()));
    if (!data.empty()) {
        const size_t got = std::fread(data.data(), 1, data.size(), f.get());
        if (got != data.size()) {
            return Err("read " + path.string() + ": short read (" + std::to_string(got) + "/" +
                       std::to_string(data.size()) + ")");
        }
    }
    return data;
}

Result<std::string> ReadTextFile(const fs::path& path) {
    auto bytes = ReadFile(path);
    if (!bytes) return Err(bytes.error());
    return std::string(bytes.value().begin(), bytes.value().end());
}

Result<std::vector<uint8_t>> ReadFileRange(const fs::path& path, uint64_t offset, size_t len) {
    File f(path, "rb");
    if (!f) return Err("open " + path.string() + ": " + Errno());
    if (std::fseek(f.get(), static_cast<long>(offset), SEEK_SET) != 0) {
        return Err("seek " + path.string() + ": " + Errno());
    }
    std::vector<uint8_t> data(len);
    const size_t got = std::fread(data.data(), 1, len, f.get());
    if (got != len) {
        return Err("read " + path.string() + " @" + std::to_string(offset) + ": short read (" +
                   std::to_string(got) + "/" + std::to_string(len) + ")");
    }
    return data;
}

Status WriteFileAtomic(const fs::path& path, std::span<const uint8_t> data) {
    const fs::path tmp = path.string() + ".tmp";
    {
        File f(tmp, "wb");
        if (!f) return Err("open " + tmp.string() + ": " + Errno());
        if (!data.empty()) {
            const size_t put = std::fwrite(data.data(), 1, data.size(), f.get());
            if (put != data.size()) {
                std::error_code ignored;
                fs::remove(tmp, ignored);
                return Err("write " + tmp.string() + ": short write");
            }
        }
        if (std::fflush(f.get()) != 0) {
            std::error_code ignored;
            fs::remove(tmp, ignored);
            return Err("flush " + tmp.string() + ": " + Errno());
        }
        if (!f.CloseChecked()) {
            std::error_code ignored;
            fs::remove(tmp, ignored);
            return Err("close " + tmp.string() + ": " + Errno());
        }
    }
    std::error_code ec;
    fs::rename(tmp, path, ec);
    if (ec) {
        std::error_code ignored;
        fs::remove(tmp, ignored);
        return Err("rename " + tmp.string() + " -> " + path.string() + ": " + ec.message());
    }
    return Status::Ok();
}

Status WriteTextFileAtomic(const fs::path& path, std::string_view text) {
    return WriteFileAtomic(
        path, std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(text.data()), text.size()));
}

}  // namespace rnet::util
