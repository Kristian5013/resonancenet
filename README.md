# ResonanceNet

**Training AI instead of hashing.**

ResonanceNet is an experimental L1 blockchain designed for decentralized training and distribution of AI.

The network trains a shared neural model using a Proof-of-Training consensus mechanism. Miners compete to improve the model's validation loss, and blocks are accepted only when the network confirms that the new checkpoint improves upon the previous state.

Each block contains a verifiable model checkpoint, allowing the blockchain to accumulate training progress and gradually develop a publicly available AI model.

Unlike centralized AI services, the model can be downloaded and run locally. Users perform inference directly on their own device — no API access, subscription, or centralized server required.

The architecture is optimized for efficient single-user inference and implementation simplicity. Compared to large transformer-based systems, the model design avoids complex optimization requirements and distributed infrastructure, making decentralized training more practical.

Early benchmarks show throughput of approximately 150,000 tokens per second on an RTX 5080-class GPU, with higher throughput expected on newer hardware.

ResonanceNet explores a different paradigm for AI infrastructure: intelligence distributed as a public artifact, not managed through centralized services.

---

## Key Features

- **Proof-of-Training (PoT)** — Miners train MinGRU neural networks and compete on validation loss improvement. No wasted computation — every block makes the model smarter.
- **Verifiable Checkpoints** — Each block includes a Keccak-256d hash of the model checkpoint. Nodes can independently verify training progress.
- **Continuous Model Growth** — The network autonomously adds layers and expands dimensions at consensus-defined heights as training progresses.
- **UTXO Expiry** — Coins expire when the network's validation loss improves 10x from their creation point. Expired funds return to mining rewards, preventing state bloat. Send-to-self resets the timer.
- **GPU Acceleration** — CUDA (full), Vulkan, Metal backends for training kernels, with CPU fallback.
- **Ed25519 Signatures** — Fast, compact, deterministic block and transaction signing.
- **Keccak-256d Hashing** — Double Keccak-256 for block hashes and transaction IDs (Ethereum-compatible, 0x01 padding).
- **Bech32 Addresses** — Human-readable `rn1...` prefix, SegWit-style P2WPKH scripts from day one.
- **Full P2P Network** — Gossip protocol, block relay, mempool synchronization, encrypted peer discovery.
- **JSON-RPC Interface** — Bitcoin Core-compatible RPC for wallets, explorers, and tooling.
- **Lightning Layer** — Payment channels for instant off-chain transactions.
- **HD Wallet** — BIP32/44 key derivation with mandatory recovery policy.

---

## How Mining Works

```
Traditional blockchain:          ResonanceNet:

  Hash(nonce) < target?            val_loss < prev_val_loss?
         |                                  |
    Pure waste                      Model improves
         |                                  |
   Block found                        Block found
```

1. Miner loads the current model checkpoint from the chain tip
2. Trains the MinGRU model on real data for N steps
3. Evaluates on a validation set
4. If `val_loss` improves over the parent block — valid block found
5. Checkpoint hash and loss are committed to the block header

The difficulty scales with model growth: as the network advances, models become larger and harder to improve.

---

## Architecture

ResonanceNet uses strict L0-L7 library layering. Higher layers depend on lower layers, never the reverse.

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
| L1 | `rnet_crypto` | Keccak-256d, Ed25519, BIP32/39 HD keys, AES-256, ChaCha20 |
| L2 | `rnet_primitives` `rnet_script` `rnet_consensus` | Blocks, transactions, script interpreter, consensus rules |
| L3 | `rnet_chain` `rnet_mempool` | UTXO set with expiry tracking, skip-list indexed mempool |
| L4 | `rnet_training` `rnet_wallet` | MinGRU training loop, HD wallet with coin selection |
| L5 | `rnet_net` `rnet_rpc` `rnet_miner` | P2P networking, JSON-RPC server, PoT mining coordinator |
| L6 | `rnet_node` `rnet_interfaces` | Node lifecycle, interface abstractions |
| L7 | `rnet_gpu` `rnet_inference` `rnet_lightning` | GPU backends, model inference, payment channels |

### Binaries

| Binary | Description |
|--------|-------------|
| `rnetd` | Full node daemon |
| `rnet-cli` | RPC command-line client |
| `rnet-tx` | Offline transaction builder |
| `rnet-wallet-tool` | Wallet creation and management |
| `rnet-util` | General-purpose utilities |
| `rnet-miner` | Standalone Proof-of-Training miner |

---

## Quick Start

### 1. Download

Pre-built binaries for Linux, macOS, and Windows are available on the [Releases](https://github.com/Kristian5013/resonancenet/releases) page.

| Platform | Archive |
|----------|---------|
| Windows x64 | `resonancenet-X.X.X-win64.zip` |
| Linux x86_64 | `resonancenet-X.X.X-x86_64-linux.tar.gz` |
| macOS ARM64 | `resonancenet-X.X.X-arm64-darwin.tar.gz` |

### 2. Extract

**Windows:**
```
# Right-click the .zip → Extract All, or:
tar -xf resonancenet-0.1.0-win64.zip
cd resonancenet-0.1.0-win64
```

> **Note:** Windows SmartScreen may show "Unknown publisher" warning on first launch.
> Click **"More info"** → **"Run anyway"**. This is normal for open-source software
> without a code signing certificate. You can verify the binaries by building from source.

**Linux / macOS:**
```bash
tar xzf resonancenet-0.1.0-x86_64-linux.tar.gz
cd resonancenet-0.1.0-x86_64-linux
```

### 3. Run a Full Node (Mainnet)

```bash
# Start the node — it will connect to seed nodes automatically
./rnetd

# On Windows:
rnetd.exe
```

The node creates a data directory at:
- **Windows**: `%APPDATA%\ResonanceNet\`
- **Linux**: `~/.resonancenet/`
- **macOS**: `~/Library/Application Support/ResonanceNet/`

### 4. Check Node Status

Open a second terminal in the same folder:

```bash
./rnet-cli getblockchaininfo
```

This shows: current block height, best block hash, validation loss, model dimensions, and sync progress.

### 5. Create a Wallet

```bash
# Create a new HD wallet with a 24-word recovery phrase
./rnet-wallet-tool create

# IMPORTANT: Write down your recovery phrase and store it safely!
# Without it, your coins are unrecoverable.

# Get your receiving address (starts with rn1...)
./rnet-cli getnewaddress
```

### 6. Start Mining (Training the AI)

Mining = training the shared neural network. You need a GPU (CUDA, Vulkan, or Metal).

```bash
# Start the miner — connects to your local node
./rnet-miner

# On Windows:
rnet-miner.exe
```

The miner will:
1. Load the latest model checkpoint from the chain tip
2. Train the MinGRU model on the consensus dataset
3. Evaluate validation loss after training
4. If loss improved by >= `difficulty_delta` → valid block found!
5. Block reward: **50 RNET** (halves as supply grows)

Expected block time: **~10 minutes** (same as Bitcoin).

### 7. Send and Receive RNET

```bash
# Check your balance
./rnet-cli getbalance

# Send coins to another address
./rnet-cli sendtoaddress rn1qxyz...destination...address 10.0

# View transaction history
./rnet-cli listtransactions
```

### 8. Lightning Network (Instant Payments)

For instant off-chain transactions:

```bash
# Open a payment channel with a peer (locks 1.0 RNET)
./rnet-cli openchannel <peer_pubkey> 1.0

# Pay a Lightning invoice instantly
./rnet-cli pay <invoice>

# Create an invoice for receiving
./rnet-cli createinvoice 0.5 "Payment for service"

# List open channels
./rnet-cli listchannels
```

### 9. Useful RPC Commands

```bash
./rnet-cli help                    # List all available commands
./rnet-cli getblockchaininfo       # Chain status, model info
./rnet-cli getmininginfo           # Mining difficulty, hashrate
./rnet-cli gettraininginfo         # Model config, loss history
./rnet-cli getpeerinfo             # Connected peers
./rnet-cli getblock <hash>         # Full block data with PoT fields
./rnet-cli getblockcount           # Current block height
./rnet-cli stop                    # Gracefully shut down the node
```

---

### Join the Testnet

```bash
# Add -testnet flag to any command
./rnetd -testnet
./rnet-cli -testnet getblockchaininfo
./rnet-miner -testnet
```

Testnet seed node: `188.137.227.180:19555`

### Local Development (regtest)

```bash
# Start a regtest node (10s block time, instant mining)
./rnetd -regtest -datadir=./data1 -port=29555 -rpcport=29554

# Generate blocks
./rnet-cli -regtest -rpcport=29554 generate 10

# Two-node test
./rnetd -regtest -datadir=./data2 -port=29565 -rpcport=29564 -connect=127.0.0.1:29555
./rnet-cli -regtest -rpcport=29554 getpeerinfo
```

---

## Building from Source

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
cmake -B build -G Ninja -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake
ninja -C build
```

---

## Network

| Network | P2P Port | RPC Port | Lightning Port | Seed Node |
|---------|----------|----------|----------------|-----------|
| Mainnet | 9555 | 9554 | 9556 | `188.137.227.180:9555` |
| Testnet | 19555 | 19554 | 19556 | `188.137.227.180:19555` |
| Regtest | 29555 | 29554 | 29556 | localhost |

## RPC API

| Method | Description |
|--------|-------------|
| `getblockchaininfo` | Chain height, best block, val_loss, model dimensions |
| `getblock <hash>` | Full block data including PoT fields |
| `gettraininginfo` | Current model config, loss history, training stats |
| `generate <n>` | Mine n blocks (regtest only) |
| `sendrawtransaction <hex>` | Broadcast a signed transaction |
| `getpeerinfo` | Connected peers and their state |
| `getmininginfo` | Mining/training status |

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-change`)
3. Follow the existing code style — C++20, no exceptions in consensus code, strict library layering
4. Ensure all tests pass (`ninja -C build && ./build/rnet_tests`)
5. Open a pull request

---

## License

ResonanceNet is released under the [MIT License](COPYING).

Copyright (c) 2025-2026 ResonanceNet Contributors

---

**Website**: [kristian5013.github.io/resonancenet](https://kristian5013.github.io/resonancenet/)
**Telegram**: [ResonanceNet AI Blockchain](https://t.me/resonance_net_main)
**Releases**: [Download](https://github.com/Kristian5013/resonancenet/releases)
**Whitepaper**: [PDF](https://kristian5013.github.io/resonancenet/ResonanceNet_Whitepaper_V2.pdf)
**Email**: pilatovichkristian2@gmail.com
