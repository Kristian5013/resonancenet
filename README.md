# ResonanceNet

**Proof-of-Training blockchain — where mining means teaching AI.**

ResonanceNet v0.1.0 is a Bitcoin Core-style blockchain built from scratch in C++20. Instead of burning electricity on hash puzzles (Proof-of-Work), miners train neural networks and submit validation loss improvements as proof. The network converges on increasingly capable AI models while securing the ledger.

52K+ lines of code. 317 source files. 15 libraries. MIT licensed.

---

## Features

- **Proof-of-Training (PoT)** — Miners train MinGRU models and compete on `val_loss` improvement. No wasted computation.
- **Keccak-256d hashing** — Double Keccak-256 for block hashes and transaction IDs.
- **Ed25519 signatures** — Fast, compact, deterministic signing.
- **Bech32 addresses** — Human-readable `rn1...` prefix (analogous to Bitcoin's `bc1`).
- **SegWit-style P2WPKH scripts** — Witness-separated transactions from day one.
- **UTXO expiry** — Coins expire when the network's `val_loss` improves 10x from their creation point. Expired funds return to mining rewards, preventing indefinite state bloat. Send-to-self resets the timer.
- **GPU acceleration** — CPU fallback, CUDA, Vulkan, and Metal backends for training kernels.
- **Lightning layer** — Payment channels on port 9556 for instant off-chain transactions.
- **Full P2P network** — Gossip protocol, block relay, mempool synchronization, peer discovery.
- **JSON-RPC interface** — Bitcoin Core-compatible RPC for wallets, explorers, and tooling.

---

## Architecture

ResonanceNet uses strict L0-L7 library layering. Higher layers may depend on lower layers, never the reverse.

```
L7  rnet_gpu, rnet_inference, rnet_lightning
L6  rnet_node, rnet_interfaces
L5  rnet_net, rnet_rpc, rnet_miner
L4  rnet_training, rnet_wallet
L3  rnet_chain, rnet_mempool
L2  rnet_primitives, rnet_script, rnet_consensus
L1  rnet_crypto
L0  rnet_util
```

| Layer | Libraries | Purpose |
|-------|-----------|---------|
| L0 | `rnet_util` | Logging, filesystem, hex encoding, serialization |
| L1 | `rnet_crypto` | Keccak-256d, Ed25519, BIP32/39 HD keys, AES-256, ChaCha20-Poly1305 |
| L2 | `rnet_primitives` `rnet_script` `rnet_consensus` | Blocks, transactions, script interpreter, consensus rules |
| L3 | `rnet_chain` `rnet_mempool` | UTXO set with expiry tracking, skip-list indexed mempool |
| L4 | `rnet_training` `rnet_wallet` | MinGRU training loop, HD wallet with coin selection |
| L5 | `rnet_net` `rnet_rpc` `rnet_miner` | P2P networking, JSON-RPC server, PoT mining coordinator |
| L6 | `rnet_node` `rnet_interfaces` | Node lifecycle, interface abstractions |
| L7 | `rnet_gpu` `rnet_inference` `rnet_lightning` | GPU backends, model inference, payment channels |

### Binaries

| Binary | Description |
|--------|-------------|
| `rnetd` | Main daemon — runs a full node |
| `rnet-cli` | RPC command-line client |
| `rnet-tx` | Offline transaction builder |
| `rnet-wallet-tool` | Wallet creation and management |
| `rnet-util` | General-purpose utilities |
| `rnet-miner` | Standalone Proof-of-Training miner |

---

## Building

### Requirements

- **C++20 compiler**: MSVC 19.50+, GCC 13+, or Clang 16+
- **CMake** 3.28+
- **Ninja** build system
- **vcpkg** for dependency management
- **Optional**: CUDA Toolkit 12+ | Vulkan SDK | Metal (macOS)

### Windows

```batch
call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=amd64
cmake -B build -G Ninja -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake
ninja -C build
```

### Linux / macOS

```bash
cmake -B build -G Ninja -DCMAKE_TOOLCHAIN_FILE=/path/to/vcpkg/scripts/buildsystems/vcpkg.cmake
ninja -C build
```

### GPU backends

GPU support is auto-detected at configure time. To force a specific backend:

```bash
cmake -B build -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE=/path/to/vcpkg/scripts/buildsystems/vcpkg.cmake \
  -DENABLE_CUDA=ON \
  -DENABLE_VULKAN=ON \
  -DENABLE_METAL=ON
```

---

## Quick Start

### Single node (regtest)

```bash
# Start a regtest node
./rnetd -regtest -datadir=./data1 -port=19555 -rpcport=19556

# Generate 10 blocks
./rnet-cli -regtest -rpcport=19556 generate 10

# Check chain state
./rnet-cli -regtest -rpcport=19556 getblockchaininfo
```

### Two-node test

```bash
# Terminal 1 — first node
./rnetd -regtest -datadir=./data1 -port=19555 -rpcport=19556

# Terminal 2 — second node, connects to first
./rnetd -regtest -datadir=./data2 -port=19565 -rpcport=19566 -connect=127.0.0.1:19555

# Verify peer connection
./rnet-cli -regtest -rpcport=19556 getpeerinfo
```

---

## RPC API

ResonanceNet exposes a JSON-RPC interface compatible with Bitcoin Core conventions.

| Method | Description |
|--------|-------------|
| `getblockchaininfo` | Chain height, best block hash, difficulty, val_loss |
| `getblock <hash>` | Full block data by hash |
| `getblockhash <height>` | Block hash at a given height |
| `generate <n>` | Mine n blocks (regtest only) |
| `sendrawtransaction <hex>` | Broadcast a signed transaction |
| `gettxout <txid> <vout>` | Query a specific UTXO |
| `getpeerinfo` | Connected peers and their state |
| `getmininginfo` | Current mining/training status |

Default RPC port: **9556** (mainnet), **19556** (regtest).

---

## Network Ports

| Network | P2P Port | RPC Port |
|---------|----------|----------|
| Mainnet | 9555 | 9556 |
| Regtest | 19555 | 19556 |

Lightning payment channels also operate on port **9556**.

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-change`)
3. Follow the existing code style — C++20, no exceptions in consensus code, strict library layering
4. Ensure all tests pass (`ninja -C build && ./build/rnet_tests`)
5. Open a pull request

Key conventions:
- `using Mutex = std::mutex;` (non-recursive) — use `*Locked()` method variants for locked access
- Coinbase outputs: Ed25519 raw `[0x20][32-byte pubkey][0xAC]`
- Regular outputs: P2WPKH `[0x00][0x14][20-byte Hash160(pubkey)]`
- Hash160 = first 20 bytes of Keccak256d(data)
- `rnet_wallet` and `rnet_node` must never import each other

---

## License

ResonanceNet is released under the [MIT License](COPYING).

Copyright (c) 2026 ResonanceNet Contributors

---

**GitHub**: [Kristian5013/resonancenet](https://github.com/Kristian5013/resonancenet)
**Website**: [kristian5013.github.io/resonancenet](https://kristian5013.github.io/resonancenet/)
**Telegram**: [resonance_net_main](https://t.me/resonance_net_main)
