# Running a model with llama.cpp

This is separate from the protocol. ResonanceNet trains a model; llama.cpp runs
one. You want it here for two reasons: to sanity-check an architecture by loading
a GGUF of something comparable, and to have a strong local assistant on the same
machine you develop on without sending code anywhere.

Everything below was run on an RTX 5090 Laptop (24 GB). Adjust the two numbers
that depend on the card — the CUDA architecture and the context length — and it
holds for anything else.

---

## Build

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build --config Release -j"$(nproc)"
```

`CMAKE_CUDA_ARCHITECTURES` matters more than it looks: omit it and CMake builds
for every architecture it knows, which turns a five-minute build into a very long
one.

| GPU | Architecture |
| --- | --- |
| RTX 5090 / 5080 / 5070 (Blackwell, incl. Laptop) | `120` |
| RTX 4090 / 4080 / 4070 (Ada) | `89` |
| RTX 3090 / 3080 (Ampere) | `86` |
| A100 | `80` |

Needs the CUDA toolkit (`nvcc --version` to check). Without one, drop
`-DGGML_CUDA=ON` and it builds a CPU-only version that works fine for small
models and is unusably slow for large ones.

The binaries land in `build/bin/`. The ones worth knowing:

| Binary | What it is for |
| --- | --- |
| `llama-server` | HTTP server with a built-in web UI. The usual way to run a model. |
| `llama-cli` | One-shot or interactive in the terminal. Good for scripts. |
| `llama-bench` | Measures prompt-processing and generation throughput. |
| `llama-quantize` | Converts an F16 GGUF to a smaller quantisation. |
| `llama-perplexity` | Measures quality loss from quantisation. |

---

## Run it

```bash
./build/bin/llama-server \
  -m ~/models/your-model.gguf \
  --host 127.0.0.1 --port 8080 \
  -ngl 99 \
  -c 32768 \
  --flash-attn on
```

Open <http://127.0.0.1:8080>.

| Flag | What it does |
| --- | --- |
| `-m` | Path to the GGUF. |
| `-ngl 99` | Offload this many layers to the GPU. 99 means all of them; lower it until the model fits. |
| `-c` | Context window in tokens. |
| `--flash-attn on` | Much less memory for attention. Leave it on. |
| `--host 127.0.0.1` | Localhost only. Use `0.0.0.0` only behind something that does authentication. |

`-ngl` and `-c` are the two knobs that decide whether it fits in VRAM, and they
compete for the same memory. The weights are fixed by the quantisation; the KV
cache grows linearly with context and with the number of offloaded layers.

If it fails to allocate, reduce `-c` first — halving the context frees far more
than dropping a couple of layers to the CPU, and layers on the CPU cost far more
speed than a shorter context costs capability.

---

## Long context

A 24 GB card running a ~30B model at Q4 has room for a large context, but not an
unbounded one. Quantise the KV cache:

```bash
./build/bin/llama-server \
  -m ~/models/your-model.gguf \
  --host 127.0.0.1 --port 8080 \
  -ngl 99 -c 1048576 \
  --flash-attn on \
  --cache-type-k q8_0 --cache-type-v q8_0
```

`q8_0` halves the cache against the default F16 at a quality cost that is hard to
detect. `q4_0` halves it again and is noticeable on long retrieval tasks. Both
require flash attention.

Two more things worth knowing when the context is very large:

- `-lm mlock` (`--load-mode mlock`) pins the model in RAM instead of letting the
  kernel page it back out. Use it if the model lives on a slow disk and the first
  token takes forever. The older `--no-mmap` and `--mlock` still work but are
  deprecated in favour of this.
- The server allocates the KV cache for the *full* declared context at startup,
  not lazily. Declaring 1M tokens and only using 8k still pays the full memory
  cost, so declare what you will actually use.

---

## One-shot and scripted use

```bash
./build/bin/llama-cli -m ~/models/your-model.gguf -ngl 99 -c 8192 \
  -p "Explain what a Merkle root proves, in three sentences." -n 256 --no-cnv
```

`--no-cnv` disables the interactive conversation loop, so it prints the completion
and exits — which is what you want in a pipeline. `-n` caps the tokens generated.

The server also speaks the OpenAI chat-completions shape, which is usually easier
to script against:

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"hello"}],"max_tokens":64}' \
  | python3 -m json.tool
```

---

## Choosing a quantisation

Rough guide for a 24 GB card:

| Quantisation | Bits/weight | A ~30B model | Quality |
| --- | --- | --- | --- |
| `Q8_0` | 8.5 | does not fit | indistinguishable from F16 |
| `Q6_K` | 6.6 | does not fit | indistinguishable in practice |
| `Q5_K_M` | 5.7 | tight | very close |
| `Q4_K_M` | 4.8 | fits | the usual choice |
| `Q4_K_XL` / `IQ4_XS` | ~4.3 | comfortable | slightly worse than Q4_K_M |
| `Q3_K_M` | 3.9 | roomy | noticeably degraded |

Below Q4 the losses stop being subtle. If a model at Q4 does not fit, a smaller
model at Q5 is almost always the better trade.

To measure rather than guess:

```bash
./build/bin/llama-perplexity -m model-Q4_K_M.gguf -f wiki.test.raw -ngl 99
```

Compare the number against the same file at F16. What matters is the difference,
not the absolute value.

---

## Benchmarking

```bash
./build/bin/llama-bench -m ~/models/your-model.gguf -ngl 99
```

Prints prompt-processing (`pp`) and token-generation (`tg`) throughput. `pp` is
how fast it reads your context; `tg` is how fast it writes. `pp` is
compute-bound and `tg` is memory-bandwidth-bound, which is why a bigger
quantisation hurts generation speed more than it hurts prompt speed.

---

## Running it detached

`nohup` alone does not survive the shell exiting. Use:

```bash
setsid nohup ./build/bin/llama-server -m ~/models/your-model.gguf \
  -ngl 99 -c 32768 --flash-attn on --port 8080 \
  > /tmp/llama.log 2>&1 < /dev/null &
```

Then `tail -f /tmp/llama.log` until it prints that the HTTP server is listening.
Stop it with `pkill -f llama-server`.
