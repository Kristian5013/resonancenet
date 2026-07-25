# Building on Unix

## Requirements

| Component | Minimum | Why |
| --- | --- | --- |
| CMake | 3.24 | build system |
| GCC | 14 | C++26 |
| Clang | 19 | C++26 (alternative to GCC) |
| Python | 3.12 | the training worker |

No external C++ libraries are required. SHA3 is vendored under `third_party/`;
everything else (JSON, argument parsing, logging, the test framework) is
implemented in-tree so a clone builds with nothing but a compiler and CMake.

### Ubuntu / Debian

```bash
sudo apt-get install build-essential cmake g++-14 python3.12 python3.12-venv
```

## Build

```bash
cmake -B build -S .
cmake --build build -j"$(nproc)"
```

Build types: `-DCMAKE_BUILD_TYPE=Release` (default), `Debug`, and
`-DWITH_SANITIZERS=ON` for address+undefined sanitizers.

## Test

```bash
ctest --test-dir build --output-on-failure     # C++ unit tests
python3 worker/tests/test_cross_language.py    # C++ <-> Python agreement
ci/run_tests.sh                                # everything CI runs
```

The cross-language test needs `build/rnet-tool`, so build first. It verifies that
the Python worker derives the same genesis hashes and the same training windows
as the C++ node — a divergence there would split the network, so it is treated as
a release blocker rather than a nicety.

## The worker

```bash
cd worker
python3 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv/bin/pip install numpy tokenizers
```

Pick the PyTorch CUDA build that matches your driver; `cu128` covers Ada and
Blackwell consumer cards.

## Sanity check

```bash
./build/rnet-tool genesis-show --network regtest
```

prints the regtest parameters and its genesis hash. If that hash differs from the
`pinned_anchor` in the same output, the build changed a consensus value — treat
it as a hard fork and do not ship it.
