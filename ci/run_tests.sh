#!/usr/bin/env bash
# Run the whole suite, with the one thing that cannot be arranged from inside it.
#
# CUBLAS_WORKSPACE_CONFIG has to be in the environment BEFORE python starts:
# torch reads it when CUDA initialises, which happens at import, so by the time
# any test could set it the decision is already made. Without it, cuBLAS picks
# workspaces by heuristic and the same batch gives different bytes on the same
# card — which is exactly the property the end-to-end tests exist to check.
#
# Without a CUDA device this is harmless, and the round tests fall back to CPU.
set -euo pipefail

cd "$(dirname "$0")/.."

export CUBLAS_WORKSPACE_CONFIG=:4096:8

PYTHON="${PYTHON:-.venv/bin/python}"
if [ ! -x "$PYTHON" ]; then
    PYTHON=python3
fi

# Say which interpreter, and refuse one that cannot import the package. The
# fallback to bare python3 used to be silent, so a machine whose system python
# had none of this installed reported eleven import errors — which reads as a
# broken tree rather than the wrong interpreter.
echo "python: $("$PYTHON" -c 'import sys; print(sys.executable, sys.version.split()[0])')"
if ! "$PYTHON" -c 'import rnet' 2>/dev/null; then
    echo "$PYTHON cannot import rnet. Install it there:" >&2
    echo "    $PYTHON -m pip install -e '.[train]'" >&2
    echo "or point this at the right one:  PYTHON=/path/to/python $0" >&2
    exit 1
fi

exec "$PYTHON" -m unittest discover -s tests -t . "$@"
