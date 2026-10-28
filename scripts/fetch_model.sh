#!/usr/bin/env bash
# Downloads the embedding model, its vocabulary, and the optional reranker into models/.
# Each file is checked against a pinned SHA-256 hash. Files that already match are kept.
#
#   all-MiniLM-L6-v2.onnx   quantized INT8 embedding model (384 dimensions), ~23 MB
#   vocab.txt               WordPiece vocabulary used by the model, ~230 KB
#   ms-marco-MiniLM-L-6-v2.onnx  quantized cross-encoder for query_reranked, ~23 MB
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/models"
mkdir -p "$DEST"

sha256_of() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | cut -d' ' -f1
    else
        shasum -a 256 "$1" | cut -d' ' -f1
    fi
}

fetch() {
    local url="$1" name="$2" expected="$3"
    local target="$DEST/$name"
    if [ -f "$target" ] && [ "$(sha256_of "$target")" = "$expected" ]; then
        echo "ok      $name"
        return
    fi
    echo "fetch   $name"
    local tmp
    tmp="$(mktemp "$DEST/.download.XXXXXX")"
    curl --fail --location --silent --show-error -o "$tmp" "$url"
    if [ "$(sha256_of "$tmp")" != "$expected" ]; then
        rm -f "$tmp"
        echo "error: $name does not match its pinned SHA-256 hash" >&2
        exit 1
    fi
    mv "$tmp" "$target"
}

HF="https://huggingface.co"
fetch "$HF/Xenova/all-MiniLM-L6-v2/resolve/main/onnx/model_quantized.onnx" \
    all-MiniLM-L6-v2.onnx \
    afdb6f1a0e45b715d0bb9b11772f032c399babd23bfc31fed1c170afc848bdb1
fetch "$HF/sentence-transformers/all-MiniLM-L6-v2/resolve/main/vocab.txt" \
    vocab.txt \
    07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3
fetch "$HF/Xenova/ms-marco-MiniLM-L-6-v2/resolve/main/onnx/model_quantized.onnx" \
    ms-marco-MiniLM-L-6-v2.onnx \
    e9d8ebf845c413e981c175bfe49a3bfa9b3dcce2a3ba54875ee5df5a58639fbe

echo "models are in $DEST"
