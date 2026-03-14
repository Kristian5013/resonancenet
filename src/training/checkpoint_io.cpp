// Copyright (c) 2024-present ResonanceNet developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://opensource.org/licenses/MIT.

// Own header.
#include "training/checkpoint_io.h"

// Project headers.
#include "crypto/keccak.h"

// Standard library.
#include <cstdio>
#include <cstring>

namespace rnet::training {

namespace {

// ===========================================================================
//  Binary I/O helpers
// ===========================================================================
// Every read/write function feeds bytes into the Keccak hasher so the final
// checksum covers the entire file content excluding itself.

// ---------------------------------------------------------------------------
// write_u32 / write_u64 / write_u8 / write_bytes
// ---------------------------------------------------------------------------
void write_u32(FILE* f, rnet::crypto::KeccakHasher& hasher, uint32_t v)
{
    uint8_t buf[4];
    buf[0] = static_cast<uint8_t>(v & 0xFF);
    buf[1] = static_cast<uint8_t>((v >> 8) & 0xFF);
    buf[2] = static_cast<uint8_t>((v >> 16) & 0xFF);
    buf[3] = static_cast<uint8_t>((v >> 24) & 0xFF);
    std::fwrite(buf, 1, 4, f);
    hasher.write({buf, 4});
}

void write_u64(FILE* f, rnet::crypto::KeccakHasher& hasher, uint64_t v)
{
    uint8_t buf[8];
    for (int i = 0; i < 8; ++i) {
        buf[i] = static_cast<uint8_t>((v >> (i * 8)) & 0xFF);
    }
    std::fwrite(buf, 1, 8, f);
    hasher.write({buf, 8});
}

void write_u8(FILE* f, rnet::crypto::KeccakHasher& hasher, uint8_t v)
{
    std::fwrite(&v, 1, 1, f);
    hasher.write({&v, 1});
}

void write_bytes(FILE* f, rnet::crypto::KeccakHasher& hasher,
                 const void* data, size_t len)
{
    std::fwrite(data, 1, len, f);
    hasher.write({static_cast<const uint8_t*>(data), len});
}

// ---------------------------------------------------------------------------
// read_u32 / read_u64 / read_u8 / read_bytes
// ---------------------------------------------------------------------------
uint32_t read_u32(FILE* f, rnet::crypto::KeccakHasher& hasher)
{
    uint8_t buf[4];
    std::fread(buf, 1, 4, f);
    hasher.write({buf, 4});
    return static_cast<uint32_t>(buf[0])
         | (static_cast<uint32_t>(buf[1]) << 8)
         | (static_cast<uint32_t>(buf[2]) << 16)
         | (static_cast<uint32_t>(buf[3]) << 24);
}

uint64_t read_u64(FILE* f, rnet::crypto::KeccakHasher& hasher)
{
    uint8_t buf[8];
    std::fread(buf, 1, 8, f);
    hasher.write({buf, 8});
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i) {
        v |= static_cast<uint64_t>(buf[i]) << (i * 8);
    }
    return v;
}

uint8_t read_u8(FILE* f, rnet::crypto::KeccakHasher& hasher)
{
    uint8_t v = 0;
    std::fread(&v, 1, 1, f);
    hasher.write({&v, 1});
    return v;
}

bool read_bytes(FILE* f, rnet::crypto::KeccakHasher& hasher,
                void* dst, size_t len)
{
    size_t got = std::fread(dst, 1, len, f);
    if (got != len) return false;
    hasher.write({static_cast<const uint8_t*>(dst), len});
    return true;
}

// ===========================================================================
//  Config serialisation
// ===========================================================================

// ---------------------------------------------------------------------------
// write_config / read_config
// ---------------------------------------------------------------------------
void write_config(FILE* f, rnet::crypto::KeccakHasher& hasher,
                  const ModelConfig& cfg)
{
    write_u32(f, hasher, cfg.d_model);
    write_u32(f, hasher, cfg.n_layers);
    write_u32(f, hasher, cfg.n_slots);
    write_u32(f, hasher, cfg.d_ff);
    write_u32(f, hasher, cfg.vocab_size);
    write_u32(f, hasher, cfg.max_seq_len);
    write_u8(f, hasher, cfg.n_conv_branches);
    for (int i = 0; i < 8; ++i) {
        write_u8(f, hasher, cfg.kernel_sizes[i]);
    }
}

ModelConfig read_config(FILE* f, rnet::crypto::KeccakHasher& hasher)
{
    ModelConfig cfg;
    cfg.d_model = read_u32(f, hasher);
    cfg.n_layers = read_u32(f, hasher);
    cfg.n_slots = read_u32(f, hasher);
    cfg.d_ff = read_u32(f, hasher);
    cfg.vocab_size = read_u32(f, hasher);
    cfg.max_seq_len = read_u32(f, hasher);
    cfg.n_conv_branches = read_u8(f, hasher);
    for (int i = 0; i < 8; ++i) {
        cfg.kernel_sizes[i] = read_u8(f, hasher);
    }
    return cfg;
}

// ===========================================================================
//  Header reading
// ===========================================================================

// ---------------------------------------------------------------------------
// read_header_impl
// ---------------------------------------------------------------------------
// Reads the common header portion (magic, version, config, step, tensor
// count), feeding every byte into the hasher.
// ---------------------------------------------------------------------------
CheckpointHeader read_header_impl(FILE* f, rnet::crypto::KeccakHasher& hasher)
{
    CheckpointHeader hdr;

    // 1. Magic.
    char magic[4];
    std::fread(magic, 1, 4, f);
    hasher.write({reinterpret_cast<const uint8_t*>(magic), 4});
    std::memcpy(hdr.magic.data(), magic, 4);

    // 2. Version.
    hdr.version = read_u32(f, hasher);

    // 3. Config.
    hdr.config = read_config(f, hasher);

    // 4. Step and tensor count.
    hdr.step = read_u64(f, hasher);
    hdr.n_tensors = read_u64(f, hasher);

    return hdr;
}

} // namespace

// ===========================================================================
//  Public API
// ===========================================================================

// ---------------------------------------------------------------------------
// read_checkpoint_header
// ---------------------------------------------------------------------------
// Reads only the header from a checkpoint file and validates the magic
// bytes and version number.
// ---------------------------------------------------------------------------
Result<CheckpointHeader> read_checkpoint_header(const std::filesystem::path& path)
{
#ifdef _WIN32
    FILE* f = _wfopen(path.c_str(), L"rb");
#else
    FILE* f = std::fopen(path.string().c_str(), "rb");
#endif
    if (!f) {
        return Result<CheckpointHeader>::err("Failed to open checkpoint file: " + path.string());
    }

    rnet::crypto::KeccakHasher hasher;
    auto hdr = read_header_impl(f, hasher);
    std::fclose(f);

    // 1. Validate magic.
    if (std::memcmp(hdr.magic.data(), "RNET", 4) != 0) {
        return Result<CheckpointHeader>::err("Invalid checkpoint magic bytes");
    }

    // 2. Validate version.
    if (hdr.version != 1) {
        return Result<CheckpointHeader>::err("Unsupported checkpoint version: " +
                                              std::to_string(hdr.version));
    }

    return Result<CheckpointHeader>::ok(std::move(hdr));
}

// ---------------------------------------------------------------------------
// read_checkpoint
// ---------------------------------------------------------------------------
// Reads the full checkpoint (header + all tensor entries) and verifies the
// trailing Keccak-256d checksum.
// ---------------------------------------------------------------------------
Result<std::vector<TensorEntry>> read_checkpoint(const std::filesystem::path& path)
{
#ifdef _WIN32
    FILE* f = _wfopen(path.c_str(), L"rb");
#else
    FILE* f = std::fopen(path.string().c_str(), "rb");
#endif
    if (!f) {
        return Result<std::vector<TensorEntry>>::err(
            "Failed to open checkpoint file: " + path.string());
    }

    rnet::crypto::KeccakHasher hasher;
    auto hdr = read_header_impl(f, hasher);

    // 1. Validate magic.
    if (std::memcmp(hdr.magic.data(), "RNET", 4) != 0) {
        std::fclose(f);
        return Result<std::vector<TensorEntry>>::err("Invalid checkpoint magic bytes");
    }

    // 2. Read tensor entries.
    std::vector<TensorEntry> tensors;
    tensors.reserve(static_cast<size_t>(hdr.n_tensors));

    constexpr size_t STREAM_BUF_SIZE = 65536;
    std::vector<uint8_t> stream_buf(STREAM_BUF_SIZE);

    for (uint64_t i = 0; i < hdr.n_tensors; ++i) {
        TensorEntry entry;

        // Name.
        uint32_t name_len = read_u32(f, hasher);
        entry.name.resize(name_len);
        if (!read_bytes(f, hasher, entry.name.data(), name_len)) {
            std::fclose(f);
            return Result<std::vector<TensorEntry>>::err(
                "Truncated checkpoint at tensor name " + std::to_string(i));
        }

        // Shape.
        uint32_t shape_dims = read_u32(f, hasher);
        entry.shape.resize(shape_dims);
        for (uint32_t d = 0; d < shape_dims; ++d) {
            uint64_t raw = read_u64(f, hasher);
            entry.shape[d] = static_cast<int64_t>(raw);
        }

        // Data (streamed in chunks for large tensors).
        uint64_t data_bytes = read_u64(f, hasher);
        entry.data.resize(static_cast<size_t>(data_bytes));

        size_t remaining = static_cast<size_t>(data_bytes);
        size_t offset = 0;
        while (remaining > 0) {
            size_t chunk = (remaining < STREAM_BUF_SIZE) ? remaining : STREAM_BUF_SIZE;
            size_t got = std::fread(entry.data.data() + offset, 1, chunk, f);
            if (got != chunk) {
                std::fclose(f);
                return Result<std::vector<TensorEntry>>::err(
                    "Truncated checkpoint at tensor data " + std::to_string(i));
            }
            hasher.write({entry.data.data() + offset, chunk});
            offset += chunk;
            remaining -= chunk;
        }

        tensors.push_back(std::move(entry));
    }

    // 3. Verify checksum.
    rnet::uint256 computed = hasher.finalize_double();

    uint8_t stored_hash[32];
    if (std::fread(stored_hash, 1, 32, f) != 32) {
        std::fclose(f);
        return Result<std::vector<TensorEntry>>::err("Truncated checkpoint checksum");
    }
    std::fclose(f);

    if (std::memcmp(computed.data(), stored_hash, 32) != 0) {
        return Result<std::vector<TensorEntry>>::err("Checkpoint checksum mismatch");
    }

    return Result<std::vector<TensorEntry>>::ok(std::move(tensors));
}

// ---------------------------------------------------------------------------
// write_checkpoint
// ---------------------------------------------------------------------------
// Writes a complete checkpoint file: header, tensor entries, and a trailing
// Keccak-256d checksum covering all preceding bytes.
// ---------------------------------------------------------------------------
Result<void> write_checkpoint(const std::filesystem::path& path,
                               const CheckpointHeader& header,
                               const std::vector<TensorEntry>& tensors)
{
#ifdef _WIN32
    FILE* f = _wfopen(path.c_str(), L"wb");
#else
    FILE* f = std::fopen(path.string().c_str(), "wb");
#endif
    if (!f) {
        return Result<void>::err("Failed to create checkpoint file: " + path.string());
    }

    rnet::crypto::KeccakHasher hasher;

    // 1. Magic.
    write_bytes(f, hasher, header.magic.data(), 4);

    // 2. Version.
    write_u32(f, hasher, header.version);

    // 3. Config.
    write_config(f, hasher, header.config);

    // 4. Step and tensor count.
    write_u64(f, hasher, header.step);
    write_u64(f, hasher, static_cast<uint64_t>(tensors.size()));

    // 5. Tensors.
    for (const auto& entry : tensors) {
        // Name.
        write_u32(f, hasher, static_cast<uint32_t>(entry.name.size()));
        write_bytes(f, hasher, entry.name.data(), entry.name.size());

        // Shape.
        write_u32(f, hasher, static_cast<uint32_t>(entry.shape.size()));
        for (auto dim : entry.shape) {
            write_u64(f, hasher, static_cast<uint64_t>(dim));
        }

        // Data (streamed in chunks).
        write_u64(f, hasher, static_cast<uint64_t>(entry.data.size()));
        constexpr size_t CHUNK_SIZE = 65536;
        size_t remaining = entry.data.size();
        size_t offset = 0;
        while (remaining > 0) {
            size_t chunk = (remaining < CHUNK_SIZE) ? remaining : CHUNK_SIZE;
            std::fwrite(entry.data.data() + offset, 1, chunk, f);
            hasher.write({entry.data.data() + offset, chunk});
            offset += chunk;
            remaining -= chunk;
        }
    }

    // 6. Checksum: keccak256d of all preceding bytes.
    rnet::uint256 checksum = hasher.finalize_double();
    std::fwrite(checksum.data(), 1, 32, f);

    std::fclose(f);
    return Result<void>::ok();
}

} // namespace rnet::training
