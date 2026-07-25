#include "consensus/genesis.h"
#include <format>

#include "crypto/sha3.h"
#include "util/fs.h"
#include "util/hex.h"

namespace rnet::consensus {

std::vector<uint8_t> GenesisContainer(Network net) { return Params(net).round.ToContainer(); }

std::string GenesisHash(Network net) {
    const auto raw = GenesisContainer(net);
    return util::ToHex(canon::ContainerId(raw));
}

Result<canon::RoundDescriptor> LoadGenesis(std::span<const uint8_t> raw,
                                           std::string_view expected_hash) {
    if (expected_hash.empty()) {
        return Err("genesis: no trust anchor pinned for this network — refusing to load");
    }
    const std::string actual = util::ToHex(canon::ContainerId(raw));
    if (actual != expected_hash) {
        return Err("genesis hash mismatch — refusing untrusted genesis\n  file:   " + actual +
                   "\n  anchor: " + std::string(expected_hash));
    }
    auto round = canon::RoundDescriptor::FromContainer(raw);
    if (!round) return Err(round.error());
    return round;
}

Result<canon::RoundDescriptor> LoadGenesisFile(const std::filesystem::path& path,
                                               std::string_view expected_hash) {
    auto raw = util::ReadFile(path);
    if (!raw) return Err(raw.error());
    return LoadGenesis(raw.value(), expected_hash);
}

Result<canon::RoundDescriptor> LoadGenesisFileForNetwork(const std::filesystem::path& path,
                                                         Network net) {
    return LoadGenesisFile(path, Params(net).genesis_hash_hex);
}

std::vector<uint8_t> PolicyContainer(Network net) { return Params(net).policy.ToContainer(); }

std::string PolicyHash(Network net) {
    return util::ToHex(canon::ContainerId(PolicyContainer(net)));
}

Result<canon::PolicyDescriptor> LoadPolicy(std::span<const uint8_t> raw,
                                           std::string_view expected_hash) {
    if (expected_hash.empty()) {
        return Err("policy: no trust anchor pinned for this network — refusing to load");
    }
    const std::string actual = util::ToHex(canon::ContainerId(raw));
    if (actual != expected_hash) {
        return Err("policy hash mismatch — refusing untrusted policy\n  file:   " + actual +
                   "\n  anchor: " + std::string(expected_hash));
    }
    return canon::PolicyDescriptor::FromContainer(raw);
}

Result<canon::PolicyDescriptor> LoadPolicyFile(const std::filesystem::path& path,
                                               std::string_view expected_hash) {
    auto raw = util::ReadFile(path);
    if (!raw) return Err(raw.error());
    return LoadPolicy(raw.value(), expected_hash);
}

Status WritePolicyFile(const std::filesystem::path& path, Network net) {
    const auto& params = Params(net);
    if (auto st = params.Validate(); !st) return st;
    return util::WriteFileAtomic(path, PolicyContainer(net));
}

Status WriteGenesisFile(const std::filesystem::path& path, Network net) {
    const auto& params = Params(net);
    if (auto st = params.Validate(); !st) return st;
    return util::WriteFileAtomic(path, GenesisContainer(net));
}

}  // namespace rnet::consensus

namespace rnet::consensus {

Result<canon::CheckpointHeader> GenesisCheckpointFor(Network net) {
    const auto& params = Params(net);

    // Refused rather than invented. A node that starts from weights it made up
    // computes deltas nobody else can reproduce, and every verification downstream
    // would fail for a reason that has nothing to do with the worker being checked.
    if (params.genesis_weights_hash_hex.empty()) {
        return Err(std::format(
            "network '{}' has no pinned genesis weights hash, so there is no agreed state to start "
            "training from. This is pinned in consensus params once the initialization procedure "
            "is fixed and its output measured; a node must not invent it.",
            params.name));
    }
    auto weights = util::FromHexFixed<32>(params.genesis_weights_hash_hex);
    if (!weights) return Err("genesis weights hash is not 32 hex-encoded bytes: " + weights.error());

    // Everything else follows from the round descriptor, so there is exactly one
    // genesis checkpoint per network and no second file to disagree with.
    canon::CheckpointHeader header;
    header.parent = crypto::Hash256{};      // genesis has no parent, by definition
    header.round_id = params.round.round_id;
    header.outer_step = 0;
    header.weights_hash = weights.value();
    header.n_params = params.round.model.ParameterCount();
    header.contributor_count = 0;
    header.weight_dtype = 1;                // bf16
    if (auto st = header.Validate(); !st) return Err(st.error());
    return header;
}

}  // namespace rnet::consensus
