// Genesis: producing and verifying the artifact that defines a network's round.
//
// A node trusts exactly one value a priori — the pinned genesis hash for its
// network — and derives everything else from an artifact that hashes to it. This
// is the same trust bootstrap as a Bitcoin node's hardcoded genesis block hash,
// and it is what stops a peer (or a local file) from feeding a node a different
// architecture than the network agreed on.
#pragma once

#include <filesystem>
#include <span>
#include <string>

#include "canon/objects.h"
#include "consensus/params.h"
#include "util/result.h"

namespace rnet::consensus {

// The canonical container bytes for a network's genesis round descriptor.
std::vector<uint8_t> GenesisContainer(Network net);

// Hex SHA3-256 of that container — the network's trust anchor.
std::string GenesisHash(Network net);

// Loads a genesis artifact and refuses it unless its hash equals `expected_hash`.
// This is the only supported way to obtain a RoundDescriptor from disk or a peer.
Result<canon::RoundDescriptor> LoadGenesis(std::span<const uint8_t> raw,
                                           std::string_view expected_hash);

Result<canon::RoundDescriptor> LoadGenesisFile(const std::filesystem::path& path,
                                               std::string_view expected_hash);

// Convenience: load the artifact for `net` and check it against the anchor pinned
// in params. Fails if the anchor has not been pinned yet.
Result<canon::RoundDescriptor> LoadGenesisFileForNetwork(const std::filesystem::path& path,
                                                         Network net);

// Writes the artifact to disk atomically. Used by rnet-tool to produce the file
// that gets distributed with releases.
Status WriteGenesisFile(const std::filesystem::path& path, Network net);

// --- the second artifact: operational policy ---
std::vector<uint8_t> PolicyContainer(Network net);
std::string PolicyHash(Network net);

Result<canon::PolicyDescriptor> LoadPolicy(std::span<const uint8_t> raw,
                                           std::string_view expected_hash);
Result<canon::PolicyDescriptor> LoadPolicyFile(const std::filesystem::path& path,
                                               std::string_view expected_hash);
Status WritePolicyFile(const std::filesystem::path& path, Network net);

// The checkpoint every node starts from: outer step 0, no parent, the weights the
// genesis round descriptor implies. Derived rather than distributed, because a
// second file stating the same fact is a second chance to disagree — and because
// the initial weights are a function of the model spec and a seed both of which
// the round descriptor already pins.
Result<canon::CheckpointHeader> GenesisCheckpointFor(Network net);

}  // namespace rnet::consensus
