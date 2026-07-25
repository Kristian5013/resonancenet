#!/usr/bin/env bash
#
# Builds a Linux release and packages it.
#
# This exists as a script rather than as instructions in a README for one reason:
# a release built by hand is a release built differently each time, and the whole
# argument for these binaries is that someone else can reproduce them. Every step
# here is a step a reader can run.
#
# What it refuses to skip:
#
#   * THE TESTS. A release that has not run its own tests is a release nobody
#     checked.
#
#   * RE-DERIVING THE ANCHORS. The genesis and policy artifacts are re-emitted
#     from the source being packaged and compared against what is compiled in. If
#     they disagree, the binaries and the artifacts they claim to implement are
#     not the same protocol, and shipping them would put that contradiction on
#     someone else's machine.
#
#   * THE INITIAL WEIGHTS. Re-derived in full for regtest. This is real
#     arithmetic over millions of parameters, not a lookup, and it is the check
#     that would catch a compiler doing something different with the
#     floating-point path.
#
# Usage: ci/build_release.sh [version]
set -euo pipefail

VERSION="${1:-dev}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BUILD_DIR=build-release
TARGET="resonancenet-${VERSION}-linux-x86_64"
DIST="dist/${TARGET}"

echo "==> configure"
# Statically linking libstdc++ and libgcc so the binary runs on a machine that
# does not have the toolchain that built it. libc stays dynamic: a fully static
# glibc binary breaks getaddrinfo, and DNS seeds are how a node finds its first
# peer.
cmake -B "$BUILD_DIR" -S . \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTS=ON \
  -DCMAKE_EXE_LINKER_FLAGS="-static-libgcc -static-libstdc++"

echo "==> build"
cmake --build "$BUILD_DIR" -j"$(nproc 2>/dev/null || echo 4)"

echo "==> tests"
ctest --test-dir "$BUILD_DIR" --output-on-failure

echo "==> consensus artifacts reproduce from this source"
mkdir -p "$BUILD_DIR/genesis-check"
for net in main test regtest; do
  out=$("$BUILD_DIR/rnet-tool" genesis-emit --network "$net" --out "$BUILD_DIR/genesis-check")
  round_emitted=$(echo "$out" | sed -n '1p' | awk '{print $2}')
  policy_emitted=$(echo "$out" | sed -n '2p' | awk '{print $2}')
  show=$("$BUILD_DIR/rnet-tool" genesis-show --network "$net")
  round_pinned=$(echo "$show" | grep '"pinned_anchor"' | cut -d'"' -f4)
  policy_pinned=$(echo "$show" | grep '"pinned_policy_anchor"' | cut -d'"' -f4)
  if [ "$round_emitted" != "$round_pinned" ] || [ "$policy_emitted" != "$policy_pinned" ]; then
    echo "FAIL: $net artifacts do not match what this binary was built to expect" >&2
    exit 1
  fi
  echo "  $net  round=$round_emitted"
done

echo "==> initial weights re-derive to the pinned anchor"
"$BUILD_DIR/rnet-tool" genesis-weights --network regtest > /dev/null

echo "==> package"
rm -rf "$DIST"
mkdir -p "$DIST"
cp "$BUILD_DIR/rnetd" "$BUILD_DIR/rnet-tool" "$DIST/"
cp COPYING README.md SECURITY.md "$DIST/"
cp -r share/genesis "$DIST/genesis"

# Recorded so a reader can tell what they have without running it. `git describe`
# is the honest answer to "which source is this", and a dirty tree says so.
{
  echo "ResonanceNet ${VERSION}"
  echo "commit:  $(git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "tree:    $(git diff --quiet 2>/dev/null && echo clean || echo DIRTY)"
  echo "built:   $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "compiler: $(${CXX:-c++} --version | head -1)"
  echo
  echo "Consensus anchors this binary was built to expect:"
  "$BUILD_DIR/rnet-tool" genesis-show --network main | grep -E 'pinned_anchor|pinned_policy_anchor|genesis_weights_hash'
} > "$DIST/BUILD-INFO.txt"

cd dist
tar czf "${TARGET}.tar.gz" "${TARGET}"
sha256sum "${TARGET}.tar.gz" > "${TARGET}.tar.gz.sha256"
cd ..

echo
echo "==> done"
ls -la "dist/${TARGET}.tar.gz" "dist/${TARGET}.tar.gz.sha256"
cat "dist/${TARGET}.tar.gz.sha256"
echo
echo "Verify with:  cd dist && sha256sum -c ${TARGET}.tar.gz.sha256"
