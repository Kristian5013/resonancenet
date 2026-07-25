#include "net/weightsstore.h"

#include <fcntl.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <format>

#include "util/hex.h"
#include "util/logging.h"

namespace rnet::net {

namespace {

// A descriptor that closes on every path out, including the error paths.
class ScopedFd {
public:
    explicit ScopedFd(int fd) : fd_(fd) {}
    ScopedFd(const ScopedFd&) = delete;
    ScopedFd& operator=(const ScopedFd&) = delete;
    ~ScopedFd() {
        if (fd_ >= 0) ::close(fd_);
    }
    int get() const { return fd_; }
    bool valid() const { return fd_ >= 0; }
    int release() {
        const int fd = fd_;
        fd_ = -1;
        return fd;
    }

private:
    int fd_;
};

// Reads exactly `out.size()` bytes at `offset`, or says how many it actually got.
// A short pread is normal at end of file and is a defect anywhere else, so the
// count is returned rather than assumed.
Result<size_t> ReadAt(int fd, uint64_t offset, std::span<uint8_t> out) {
    size_t filled = 0;
    while (filled < out.size()) {
        const ssize_t got =
            ::pread(fd, out.data() + filled, out.size() - filled, static_cast<off_t>(offset + filled));
        if (got < 0) {
            if (errno == EINTR) continue;
            return Err(std::string("weights: pread: ") + std::strerror(errno));
        }
        if (got == 0) break;   // end of file
        filled += static_cast<size_t>(got);
    }
    return filled;
}

}  // namespace

WeightsStore::WeightsStore(std::filesystem::path directory) : directory_(std::move(directory)) {
    std::error_code ec;
    std::filesystem::create_directories(directory_, ec);
    if (ec) {
        RNET_LOG_ERROR("weights: cannot create {}: {}", directory_.string(), ec.message());
    }
}

std::filesystem::path WeightsStore::PathFor(const crypto::Hash256& hash) const {
    return directory_ / (util::ToHex(hash) + ".weights");
}

Status WeightsStore::PutFromFd(const crypto::Hash256& hash, int fd, uint64_t expected_bytes) {
    if (expected_bytes == 0) return Err("weights: refusing an empty checkpoint");
    if (Has(hash)) return Status::Ok();   // already held; content-addressed, so identical

    // Written under a temporary name and renamed only once the hash matches. A node
    // killed mid-write must leave either nothing or a complete file — a truncated
    // file that kept its final name would be a checkpoint whose bytes silently
    // disagree with the hash they are stored under.
    const auto final_path = PathFor(hash);
    const auto temp_path = directory_ / (util::ToHex(hash) + ".partial");

    ScopedFd out(::open(temp_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600));
    if (!out.valid()) {
        return Err(std::format("weights: cannot create {}: {}", temp_path.string(),
                               std::strerror(errno)));
    }

    crypto::Sha3Hasher hasher;
    std::vector<uint8_t> slice(kWeightsHashSlice);
    uint64_t copied = 0;
    while (copied < expected_bytes) {
        const size_t want =
            static_cast<size_t>(std::min<uint64_t>(slice.size(), expected_bytes - copied));
        auto got = ReadAt(fd, copied, std::span<uint8_t>(slice.data(), want));
        if (!got) {
            std::error_code ec;
            std::filesystem::remove(temp_path, ec);
            return Err(got.error());
        }
        if (got.value() == 0) {
            std::error_code ec;
            std::filesystem::remove(temp_path, ec);
            return Err(std::format("weights: source ended after {} of {} bytes", copied,
                                   expected_bytes));
        }
        hasher.Update(std::span<const uint8_t>(slice.data(), got.value()));

        size_t written = 0;
        while (written < got.value()) {
            const ssize_t n = ::write(out.get(), slice.data() + written, got.value() - written);
            if (n < 0) {
                if (errno == EINTR) continue;
                std::error_code ec;
                std::filesystem::remove(temp_path, ec);
                return Err(std::string("weights: write: ") + std::strerror(errno));
            }
            written += static_cast<size_t>(n);
        }
        copied += got.value();
    }

    // Verified before it is given the name it will be served under. A node that
    // served weights it had not checked would spread a model nobody agreed on,
    // under a name everybody trusts.
    const auto actual = hasher.Finalize();
    if (actual != hash) {
        std::error_code ec;
        std::filesystem::remove(temp_path, ec);
        return Err(std::format("weights: bytes hash to {}, not the claimed {}",
                               util::ToHex(actual), util::ToHex(hash)));
    }
    if (::fsync(out.get()) != 0) {
        RNET_LOG_WARN("weights: fsync failed: {}", std::strerror(errno));
    }

    std::error_code ec;
    std::filesystem::rename(temp_path, final_path, ec);
    if (ec) {
        std::filesystem::remove(temp_path, ec);
        return Err(std::format("weights: cannot publish {}: {}", final_path.string(),
                               ec.message()));
    }
    RNET_LOG_INFO("weights: stored {} ({} bytes)", util::ToHex(hash).substr(0, 16), expected_bytes);
    return Status::Ok();
}

Status WeightsStore::Put(const crypto::Hash256& hash, std::span<const uint8_t> bytes) {
    if (bytes.empty()) return Err("weights: refusing an empty checkpoint");
    if (crypto::Sha3_256(bytes) != hash) {
        return Err("weights: bytes do not hash to the claimed value");
    }
    if (Has(hash)) return Status::Ok();

    const auto final_path = PathFor(hash);
    const auto temp_path = directory_ / (util::ToHex(hash) + ".partial");
    ScopedFd out(::open(temp_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600));
    if (!out.valid()) {
        return Err(std::format("weights: cannot create {}: {}", temp_path.string(),
                               std::strerror(errno)));
    }
    size_t written = 0;
    while (written < bytes.size()) {
        const ssize_t n = ::write(out.get(), bytes.data() + written, bytes.size() - written);
        if (n < 0) {
            if (errno == EINTR) continue;
            std::error_code ec;
            std::filesystem::remove(temp_path, ec);
            return Err(std::string("weights: write: ") + std::strerror(errno));
        }
        written += static_cast<size_t>(n);
    }
    std::error_code ec;
    std::filesystem::rename(temp_path, final_path, ec);
    if (ec) {
        std::filesystem::remove(temp_path, ec);
        return Err(std::format("weights: cannot publish {}: {}", final_path.string(),
                               ec.message()));
    }
    return Status::Ok();
}

bool WeightsStore::Has(const crypto::Hash256& hash) const {
    std::error_code ec;
    return std::filesystem::is_regular_file(PathFor(hash), ec);
}

Result<uint64_t> WeightsStore::SizeOf(const crypto::Hash256& hash) const {
    std::error_code ec;
    const auto size = std::filesystem::file_size(PathFor(hash), ec);
    if (ec) return Err(std::format("weights: {} is not held", util::ToHex(hash).substr(0, 16)));
    return static_cast<uint64_t>(size);
}

Result<size_t> WeightsStore::ReadRange(const crypto::Hash256& hash, uint64_t offset,
                                       std::span<uint8_t> out) const {
    ScopedFd fd(::open(PathFor(hash).c_str(), O_RDONLY | O_CLOEXEC));
    if (!fd.valid()) {
        return Err(std::format("weights: {} is not held", util::ToHex(hash).substr(0, 16)));
    }
    return ReadAt(fd.get(), offset, out);
}

Result<int> WeightsStore::OpenForRead(const crypto::Hash256& hash) const {
    const int fd = ::open(PathFor(hash).c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        return Err(std::format("weights: {} is not held", util::ToHex(hash).substr(0, 16)));
    }
    return fd;
}

Result<std::vector<crypto::Hash256>> WeightsStore::List() const {
    std::vector<crypto::Hash256> held;
    std::error_code ec;
    for (const auto& entry : std::filesystem::directory_iterator(directory_, ec)) {
        if (ec) break;
        if (!entry.is_regular_file()) continue;
        const auto name = entry.path().filename().string();
        if (!name.ends_with(".weights")) continue;
        auto hash = util::FromHexFixed<32>(name.substr(0, name.size() - 8));
        if (!hash) continue;   // not one of ours; left alone rather than deleted
        held.push_back(hash.value());
    }
    return held;
}

Result<uint64_t> WeightsStore::TotalBytes() const {
    uint64_t total = 0;
    std::error_code ec;
    for (const auto& entry : std::filesystem::directory_iterator(directory_, ec)) {
        if (ec) break;
        if (entry.is_regular_file()) total += entry.file_size(ec);
    }
    return total;
}

Result<size_t> WeightsStore::RetainOnly(const std::set<crypto::Hash256>& keep) {
    auto held = List();
    if (!held) return Err(held.error());

    size_t removed = 0;
    for (const auto& hash : held.value()) {
        if (keep.find(hash) != keep.end()) continue;
        std::error_code ec;
        if (std::filesystem::remove(PathFor(hash), ec)) {
            ++removed;
            RNET_LOG_INFO("weights: pruned {} (outside the retention window)",
                          util::ToHex(hash).substr(0, 16));
        }
    }

    // A partial file is a write that never finished. Nothing refers to it, so it is
    // dead weight rather than a checkpoint anyone could ask for.
    std::error_code ec;
    for (const auto& entry : std::filesystem::directory_iterator(directory_, ec)) {
        if (ec) break;
        if (entry.is_regular_file() && entry.path().extension() == ".partial") {
            std::error_code remove_ec;
            std::filesystem::remove(entry.path(), remove_ec);
        }
    }
    return removed;
}

}  // namespace rnet::net
