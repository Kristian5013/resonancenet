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

exec "$PYTHON" -m unittest discover -s tests -t . "$@"
