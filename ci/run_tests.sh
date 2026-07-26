#!/usr/bin/env bash
#
# Portable CI entry point. Runs the same checks locally and in GitHub Actions, so
# "works on my machine" and "passes CI" cannot diverge.
#
#   ci/run_tests.sh            release build + tests
#   ci/run_tests.sh sanitize   address+undefined sanitizers
set -euo pipefail

MODE="${1:-release}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

case "$MODE" in
  release)  BUILD_DIR=build-ci;      CMAKE_ARGS=(-DCMAKE_BUILD_TYPE=Release) ;;
  debug)    BUILD_DIR=build-ci-dbg;  CMAKE_ARGS=(-DCMAKE_BUILD_TYPE=Debug) ;;
  sanitize) BUILD_DIR=build-ci-asan; CMAKE_ARGS=(-DCMAKE_BUILD_TYPE=Debug -DWITH_SANITIZERS=ON) ;;
  *) echo "usage: $0 [release|debug|sanitize]" >&2; exit 2 ;;
esac

echo "==> configure ($MODE)"
cmake -B "$BUILD_DIR" -S . "${CMAKE_ARGS[@]}"

echo "==> build"
cmake --build "$BUILD_DIR" -j"$(nproc 2>/dev/null || echo 4)"

# A test that disappears takes its proof with it and leaves the fix looking just
# as covered as before. That happened here: three tests vanished when a whole file
# was restored over them, and it was caught only because someone noticed the count
# drop by three. A count says something died; the list says which.
echo "==> registered tests"
"$BUILD_DIR/rnet_tests" --list > "$BUILD_DIR/registered_tests.txt"
if ! diff -u test/registered_tests.txt "$BUILD_DIR/registered_tests.txt"; then
  cat >&2 <<'MSG'

The set of registered tests does not match test/registered_tests.txt.

  Lines marked - are tests that exist in the file and NOT in the binary. Those
  were deleted or renamed. If that was not deliberate, restore them.

  Lines marked + are new tests. Record them:
      ./build/rnet_tests --list > test/registered_tests.txt
MSG
  exit 1
fi

echo "==> unit tests"
ctest --test-dir "$BUILD_DIR" --output-on-failure

# Both artifacts are consensus values. Re-emitting must reproduce exactly what is
# compiled in; a mismatch means an accidental fork of the model or of the policy.
echo "==> genesis and policy reproducibility"
for net in main test regtest; do
  out=$("$BUILD_DIR/rnet-tool" genesis-emit --network "$net" --out "$BUILD_DIR/genesis")
  round_emitted=$(echo "$out" | sed -n '1p' | awk '{print $2}')
  policy_emitted=$(echo "$out" | sed -n '2p' | awk '{print $2}')
  show=$("$BUILD_DIR/rnet-tool" genesis-show --network "$net")
  round_pinned=$(echo "$show" | grep '"pinned_anchor"' | cut -d'"' -f4)
  policy_pinned=$(echo "$show" | grep '"pinned_policy_anchor"' | cut -d'"' -f4)
  if [ "$round_emitted" != "$round_pinned" ]; then
    echo "FAIL: $net round $round_emitted != pinned $round_pinned" >&2
    exit 1
  fi
  if [ "$policy_emitted" != "$policy_pinned" ]; then
    echo "FAIL: $net policy $policy_emitted != pinned $policy_pinned" >&2
    exit 1
  fi
  echo "  $net round=$round_emitted"
  echo "  $net policy=$policy_emitted"
done

# Where training starts. Nodes that begin from different weights compute deltas
# that cannot be aggregated and cannot be verified, and the failure looks like
# every worker cheating at once. Regtest is re-derived in full on every run;
# re-deriving 1B parameters for main takes minutes, so it is checked when
# RNET_CHECK_ALL_WEIGHTS is set (release builds, nightly).
echo "==> initial weights"
NETS_TO_DERIVE="regtest"
if [ -n "${RNET_CHECK_ALL_WEIGHTS:-}" ]; then
  NETS_TO_DERIVE="main test regtest"
fi
for net in $NETS_TO_DERIVE; do
  if ! "$BUILD_DIR/rnet-tool" genesis-weights --network "$net" > /dev/null; then
    echo "FAIL: $net derived weights do not match the pinned anchor" >&2
    exit 1
  fi
  echo "  $net weights match the pinned anchor"
done

# The C++ node and the Python worker must agree byte for byte, or workers and
# verifiers would disagree about what was trained.
if command -v python3 > /dev/null; then
  echo "==> cross-language agreement"
  python3 worker/tests/test_cross_language.py
else
  echo "==> cross-language agreement (skipped: no python3)"
fi

# The local channel, against the real daemon binary rather than a reimplementation
# — the same reason the other cross-language checks drive rnet-tool.
if command -v python3 > /dev/null; then
  echo "==> worker to daemon channel"
  python3 worker/tests/test_ipc_roundtrip.py
else
  echo "==> worker to daemon channel (skipped: no python3)"
fi

# The protocol end to end on one machine: genesis, schedule, training,
# aggregation, replay, bit-exact comparison. Needs a GPU and the worker venv, so
# it is skipped rather than failed where those are absent.
if [ -x worker/.venv/bin/python ]; then
  echo "==> local protocol simulation"
  worker/.venv/bin/python worker/tests/test_local_simulation.py --scaling 2
else
  echo "==> local protocol simulation (skipped: worker/.venv not present)"
fi

echo "==> all checks passed"
