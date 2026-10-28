#!/usr/bin/env bash
# Builds the C ABI, compiles the C integration test and the C example against it, and runs
# both. Needs gcc, cargo, and the model files (run scripts/fetch_model.sh first).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
MODEL="${RAGLITE_MODEL:-$ROOT/models/all-MiniLM-L6-v2.onnx}"
if [ ! -f "$MODEL" ]; then
    echo "model not found at $MODEL; run scripts/fetch_model.sh" >&2
    exit 1
fi

cargo build --release -p tpt-rag-ffi --manifest-path "$ROOT/Cargo.toml"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

CFLAGS="-Wall -Wextra -Werror -I$ROOT"
LIBS="-L$ROOT/target/release -ltpt_rag_ffi"
gcc $CFLAGS "$ROOT/tpt-rag-ffi/tests/c_integration.c" $LIBS -o "$WORK/c_integration"
gcc $CFLAGS "$ROOT/examples/c_basic.c" $LIBS -o "$WORK/c_basic"

export RAGLITE_MODEL="$MODEL"
export LD_LIBRARY_PATH="$ROOT/target/release${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

mkdir -p "$WORK/integration"
"$WORK/c_integration" "$WORK/integration"
(cd "$WORK" && "$WORK/c_basic")
echo "C ABI tests passed"
