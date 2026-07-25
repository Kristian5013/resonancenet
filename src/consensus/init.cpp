#include "consensus/init.h"

#include <cmath>
#include <cstring>
#include <format>

#include "canon/canon.h"

namespace rnet::consensus {

namespace {

// Domain separation. Without it, the same descriptor bytes could be made to
// produce the same stream for a different purpose somewhere else in the protocol.
constexpr char kInitDomain[] = "rnet/init/v1";

// How many elements are generated per SHA3 block. Each 32-byte digest yields
// eight 32-bit words, and each word yields one value.
constexpr uint32_t kValuesPerBlock = 8;

uint32_t BeU32(const uint8_t* p) {
    return (static_cast<uint32_t>(p[0]) << 24) | (static_cast<uint32_t>(p[1]) << 16) |
           (static_cast<uint32_t>(p[2]) << 8) | static_cast<uint32_t>(p[3]);
}

// One block of the counter-based stream. Nothing depends on the blocks before it,
// so any position is reachable directly.
crypto::Hash256 StreamBlock(const crypto::Hash256& seed, uint32_t tensor_index, uint64_t counter) {
    canon::Writer w;
    w.Hash(seed);
    w.U32(tensor_index);
    w.U64(counter);
    return crypto::Sha3_256(w.data());
}

// A uniform value in [-bound, bound].
//
// Every operation here is exact or correctly rounded by IEEE-754:
//   * `word >> 8` keeps 24 bits, which converts to float exactly (a float32
//     mantissa holds 24 bits), and the multiply by 2^-24 is exact — it only
//     changes the exponent.
//   * `2x - 1` is exact for the same reason.
//   * the final multiply is one correctly-rounded operation.
// No implementation-defined library function appears, so two languages that both
// follow IEEE-754 produce identical bits.
float UniformSymmetric(uint32_t word, float bound) {
    const float unit = static_cast<float>(word >> 8) * 0x1p-24f;   // [0, 1)
    return (2.0f * unit - 1.0f) * bound;
}

}  // namespace

uint16_t FloatToBf16(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));

    // NaN must stay NaN rather than becoming infinity through rounding, which is
    // what a plain truncate-and-round would do to a NaN whose payload is only in
    // the low bits.
    if ((bits & 0x7F800000u) == 0x7F800000u && (bits & 0x007FFFFFu) != 0) {
        return static_cast<uint16_t>((bits >> 16) | 0x0040u);
    }
    // Round half to even: add the rounding bias, then truncate.
    const uint32_t lsb = (bits >> 16) & 1u;
    const uint32_t rounding_bias = 0x7FFFu + lsb;
    return static_cast<uint16_t>((bits + rounding_bias) >> 16);
}

std::vector<TensorSpec> TensorLayout(const canon::ModelSpec& model) {
    std::vector<TensorSpec> tensors;
    const uint64_t d = model.d_model;
    const uint64_t head_dim = d / model.n_heads;
    const uint64_t q_dim = static_cast<uint64_t>(model.n_heads) * head_dim;
    const uint64_t kv_dim = static_cast<uint64_t>(model.n_kv_heads) * head_dim;

    // Embeddings first, then layers in order, then the final norm and head. This
    // ordering is consensus: the weights hash is over the concatenation.
    tensors.push_back({"tok_embeddings", model.vocab_size, d, false});

    for (uint32_t layer = 0; layer < model.n_layers; ++layer) {
        const auto name = [&](const char* suffix) {
            return std::format("layers.{}.{}", layer, suffix);
        };
        tensors.push_back({name("attention_norm"), d, 1, true});
        tensors.push_back({name("attention.wq"), q_dim, d, false});
        tensors.push_back({name("attention.wk"), kv_dim, d, false});
        tensors.push_back({name("attention.wv"), kv_dim, d, false});
        tensors.push_back({name("attention.wo"), d, q_dim, false});
        if (model.qk_norm) {
            tensors.push_back({name("attention.q_norm"), head_dim, 1, true});
            tensors.push_back({name("attention.k_norm"), head_dim, 1, true});
        }
        tensors.push_back({name("ffn_norm"), d, 1, true});
        tensors.push_back({name("feed_forward.w_gate"), model.d_ff, d, false});
        tensors.push_back({name("feed_forward.w_up"), model.d_ff, d, false});
        tensors.push_back({name("feed_forward.w_down"), d, model.d_ff, false});
    }

    tensors.push_back({"norm", d, 1, true});
    // A tied head is the embedding matrix read the other way round; emitting it
    // again would both double the storage and let the two copies drift.
    if (!model.tie_embeddings) tensors.push_back({"output", model.vocab_size, d, false});
    return tensors;
}

crypto::Hash256 InitSeed(const canon::RoundDescriptor& round) {
    canon::Writer w;
    w.String(kInitDomain);
    // The whole descriptor, through its canonical container: the seed then depends
    // on every consensus value the round fixes, including the network magic. Two
    // networks with the same architecture start from different weights.
    w.Hash(round.Id());
    return crypto::Sha3_256(w.data());
}

Status InitTensorSlice(const crypto::Hash256& seed, uint32_t tensor_index,
                       const TensorSpec& tensor, uint64_t element_offset, std::span<float> out) {
    if (element_offset + out.size() > tensor.elements()) {
        return Err(std::format("init: slice [{}, {}) exceeds tensor '{}' of {} elements",
                               element_offset, element_offset + out.size(), tensor.name,
                               tensor.elements()));
    }
    if (tensor.is_norm) {
        // Norm weights start at one. A random norm scale at step zero would make
        // the first forward pass depend on noise in the one place the model has no
        // capacity to correct it.
        std::fill(out.begin(), out.end(), 1.0f);
        return Status::Ok();
    }
    if (tensor.cols == 0) return Err("init: tensor has zero fan-in");

    // 1/sqrt(fan_in). Division and square root are both correctly rounded by
    // IEEE-754, so this is the same float on every conforming platform.
    const float bound = 1.0f / std::sqrt(static_cast<float>(tensor.cols));

    uint64_t position = element_offset;
    size_t written = 0;
    while (written < out.size()) {
        const uint64_t block_index = position / kValuesPerBlock;
        const uint32_t within = static_cast<uint32_t>(position % kValuesPerBlock);
        const auto block = StreamBlock(seed, tensor_index, block_index);

        for (uint32_t i = within; i < kValuesPerBlock && written < out.size(); ++i) {
            out[written++] = UniformSymmetric(BeU32(block.data() + i * 4), bound);
            ++position;
        }
    }
    return Status::Ok();
}

Result<crypto::Hash256> GenesisWeightsHash(const canon::RoundDescriptor& round) {
    if (auto st = round.model.Validate(); !st) return Err(st.error());

    const auto layout = TensorLayout(round.model);
    const auto seed = InitSeed(round);

    // Cross-checked against the independently written parameter count. If the two
    // disagree, one of them is wrong about the model — and shipping either would
    // mean the network's idea of its own size does not match its weights.
    uint64_t total = 0;
    for (const auto& tensor : layout) total += tensor.elements();
    if (total != round.model.ParameterCount()) {
        return Err(std::format(
            "init: the tensor layout holds {} parameters but the model spec says {} — the two "
            "descriptions of the same model disagree",
            total, round.model.ParameterCount()));
    }

    // Streamed in bounded memory: a 1B-parameter model must be hashable on a
    // machine that cannot hold it.
    constexpr size_t kSliceElements = 1u << 20;
    std::vector<float> values(kSliceElements);
    std::vector<uint8_t> bytes(kSliceElements * 2);

    crypto::Sha3Hasher hasher;
    for (uint32_t index = 0; index < layout.size(); ++index) {
        const auto& tensor = layout[index];
        uint64_t done = 0;
        while (done < tensor.elements()) {
            const size_t count =
                static_cast<size_t>(std::min<uint64_t>(kSliceElements, tensor.elements() - done));
            if (auto st = InitTensorSlice(seed, index, tensor, done,
                                          std::span<float>(values.data(), count));
                !st) {
                return Err(st.error());
            }
            for (size_t i = 0; i < count; ++i) {
                const uint16_t bf16 = FloatToBf16(values[i]);
                bytes[i * 2] = static_cast<uint8_t>(bf16 >> 8);   // big-endian, as everywhere
                bytes[i * 2 + 1] = static_cast<uint8_t>(bf16 & 0xFF);
            }
            hasher.Update(std::span<const uint8_t>(bytes.data(), count * 2));
            done += count;
        }
    }
    return hasher.Finalize();
}

}  // namespace rnet::consensus
