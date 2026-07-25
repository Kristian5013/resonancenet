// The weights a checkpoint refers to.
//
// A CheckpointHeader carries a hash and no bytes. That is right for the chain —
// links are 32 bytes and a node can follow the whole history without holding any
// model — but it means a node that only holds headers cannot answer the one
// question a joining worker asks: "give me the weights for the tip so I can start
// from where everyone else is."
//
// Until this existed, a network of one node worked and a second node could never
// join. It could derive the genesis weights, but it could not catch up, because
// catching up needs either every historical payload (which the retention window
// deliberately does not keep) or the weights themselves.
//
// Three properties, each forced by the size of the thing:
//
//   * ON DISK, NOT IN MEMORY. RN-1B in bf16 is 1.97 GB per checkpoint and the
//     policy retains several. Holding those in RAM alongside a training process
//     that already wants 23 GB is not a tuning problem, it is a design error.
//
//   * CONTENT-ADDRESSED AND VERIFIED ON ADMISSION. The file is named by the hash
//     of its contents, and nothing is admitted until the bytes have been hashed
//     and matched. A node that served weights it had not verified would spread a
//     model nobody agreed on, under a name everybody trusts.
//
//   * WRITTEN ATOMICALLY. A node killed mid-write must leave either the old file
//     or nothing — never a truncated one, because a truncated file that keeps its
//     name is a checkpoint whose bytes silently disagree with its hash.
//
// Retention is the policy's, not this class's: it prunes to whatever set it is
// told to keep, so the rule about how much history exists lives in one place.
#pragma once

#include <cstdint>
#include <filesystem>
#include <set>
#include <span>
#include <vector>

#include "crypto/sha3.h"
#include "util/result.h"

namespace rnet::net {

// Read in slices while hashing, so admitting a two-gigabyte file never needs two
// gigabytes of memory.
inline constexpr size_t kWeightsHashSlice = 8u * 1024 * 1024;

class WeightsStore {
public:
    // `directory` is created if absent. Nothing is read at construction: a store
    // pointing at a directory full of files discovers them lazily, so a corrupt
    // leftover costs nothing until something actually asks for it.
    explicit WeightsStore(std::filesystem::path directory);

    // Admits bytes read from `fd`, which must hold exactly `expected_bytes`.
    // Hashes while copying and refuses anything that does not match `hash` — the
    // sender's claim about its own bytes is never the reason they are stored.
    Status PutFromFd(const crypto::Hash256& hash, int fd, uint64_t expected_bytes);

    // Admits bytes already in memory. Used where the caller assembled them itself,
    // such as a bulk transfer that completed over the network.
    Status Put(const crypto::Hash256& hash, std::span<const uint8_t> bytes);

    bool Has(const crypto::Hash256& hash) const;
    Result<uint64_t> SizeOf(const crypto::Hash256& hash) const;

    // Reads a slice. This is how weights are served to a peer chunk by chunk
    // without the whole model being resident.
    Result<size_t> ReadRange(const crypto::Hash256& hash, uint64_t offset,
                             std::span<uint8_t> out) const;

    // A read-only descriptor, for handing to a local worker. The worker verifies
    // the contents against the checkpoint's hash before using them, which is what
    // makes passing an ordinary file descriptor safe in this direction: the
    // receiver checks, so the sender does not have to be trusted.
    Result<int> OpenForRead(const crypto::Hash256& hash) const;

    // Deletes everything not named. The set comes from the checkpoint chain's
    // retention window, so the rule about how much history exists stays in one
    // place instead of being duplicated here.
    //
    // Returns how many files were removed.
    Result<size_t> RetainOnly(const std::set<crypto::Hash256>& keep);

    // What is held, for status and for tests.
    Result<std::vector<crypto::Hash256>> List() const;
    Result<uint64_t> TotalBytes() const;

    const std::filesystem::path& directory() const { return directory_; }

private:
    std::filesystem::path PathFor(const crypto::Hash256& hash) const;

    std::filesystem::path directory_;
};

}  // namespace rnet::net
