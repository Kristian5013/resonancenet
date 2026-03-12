#!/bin/bash
set -euo pipefail

PREFIX="${PREFIX:-/usr/local}"
DATADIR="/var/lib/resonancenet"
CONFDIR="/etc/resonancenet"

echo "=== ResonanceNet Linux Installer ==="

# Check if built
if [ ! -f build/rnetd ]; then
    echo "Error: build/rnetd not found. Run 'ninja -C build' first."
    exit 1
fi

# Create user
if ! id -u rnet &>/dev/null; then
    echo "Creating user 'rnet'..."
    sudo useradd -r -m -d "$DATADIR" -s /usr/sbin/nologin rnet
fi

# Install binaries
echo "Installing binaries to $PREFIX/bin..."
sudo install -m 0755 build/rnetd "$PREFIX/bin/"
sudo install -m 0755 build/rnet-cli "$PREFIX/bin/" 2>/dev/null || true
sudo install -m 0755 build/rnet-tx "$PREFIX/bin/" 2>/dev/null || true
sudo install -m 0755 build/rnet-wallet-tool "$PREFIX/bin/" 2>/dev/null || true
sudo install -m 0755 build/rnet-util "$PREFIX/bin/" 2>/dev/null || true
sudo install -m 0755 build/rnet-miner "$PREFIX/bin/" 2>/dev/null || true

# Create directories
sudo mkdir -p "$DATADIR" "$CONFDIR"
sudo chown rnet:rnet "$DATADIR"
sudo chmod 710 "$DATADIR"

# Default config
if [ ! -f "$CONFDIR/resonancenet.conf" ]; then
    echo "Creating default config..."
    sudo tee "$CONFDIR/resonancenet.conf" > /dev/null << 'CONF'
# ResonanceNet configuration
# See: rnetd -help for all options

# Network
# testnet=1
# regtest=1

# RPC
rpcport=9556
rpcbind=127.0.0.1

# P2P
port=9555
listen=1
maxconnections=125

# Mining (Proof-of-Training)
# mine=1
# gpubackend=cuda  # cpu, cuda, vulkan, metal
CONF
    sudo chown root:rnet "$CONFDIR/resonancenet.conf"
    sudo chmod 640 "$CONFDIR/resonancenet.conf"
fi

# Install systemd service
if [ -d /etc/systemd/system ]; then
    echo "Installing systemd service..."
    sudo cp contrib/init/rnetd.service /etc/systemd/system/
    sudo systemctl daemon-reload
    echo "  Enable with: sudo systemctl enable rnetd"
    echo "  Start with:  sudo systemctl start rnetd"
fi

echo ""
echo "=== Installation complete ==="
echo "  Binaries:  $PREFIX/bin/rnetd, rnet-cli, ..."
echo "  Data dir:  $DATADIR"
echo "  Config:    $CONFDIR/resonancenet.conf"
echo "  Service:   sudo systemctl start rnetd"
