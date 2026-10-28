# Downloads the embedding model, its vocabulary, and the optional reranker into models/.
# Each file is checked against a pinned SHA-256 hash. Files that already match are kept.
#
#   all-MiniLM-L6-v2.onnx        quantized INT8 embedding model (384 dimensions), ~23 MB
#   vocab.txt                    WordPiece vocabulary used by the model, ~230 KB
#   ms-marco-MiniLM-L-6-v2.onnx  quantized cross-encoder for query_reranked, ~23 MB
$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $PSScriptRoot
$dest = Join-Path $root 'models'
New-Item -ItemType Directory -Force -Path $dest | Out-Null

function Fetch-Model([string]$url, [string]$name, [string]$expected) {
    $target = Join-Path $dest $name
    if ((Test-Path $target) -and ((Get-FileHash $target -Algorithm SHA256).Hash.ToLower() -eq $expected)) {
        Write-Output "ok      $name"
        return
    }
    Write-Output "fetch   $name"
    $tmp = "$target.download"
    Invoke-WebRequest -Uri $url -OutFile $tmp -UseBasicParsing
    if ((Get-FileHash $tmp -Algorithm SHA256).Hash.ToLower() -ne $expected) {
        Remove-Item $tmp -Force
        throw "$name does not match its pinned SHA-256 hash"
    }
    Move-Item $tmp $target -Force
}

$hf = 'https://huggingface.co'
Fetch-Model "$hf/Xenova/all-MiniLM-L6-v2/resolve/main/onnx/model_quantized.onnx" `
    'all-MiniLM-L6-v2.onnx' 'afdb6f1a0e45b715d0bb9b11772f032c399babd23bfc31fed1c170afc848bdb1'
Fetch-Model "$hf/sentence-transformers/all-MiniLM-L6-v2/resolve/main/vocab.txt" `
    'vocab.txt' '07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3'
Fetch-Model "$hf/Xenova/ms-marco-MiniLM-L-6-v2/resolve/main/onnx/model_quantized.onnx" `
    'ms-marco-MiniLM-L-6-v2.onnx' 'e9d8ebf845c413e981c175bfe49a3bfa9b3dcce2a3ba54875ee5df5a58639fbe'

Write-Output "models are in $dest"
