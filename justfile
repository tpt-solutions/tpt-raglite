# Task runner for tpt-raglite. Install with `cargo install just`, then run `just <target>`.

default:
    @just --list

# Download the embedding and reranker models into models/.
setup:
    bash scripts/fetch_model.sh

# Format, lint, and run every Rust test.
test:
    cargo fmt --all -- --check
    cargo clippy --workspace --all-targets -- -D warnings
    cargo test --workspace

# Build and run the C ABI tests.
ffi-test:
    bash tpt-rag-ffi/tests/run_c_integration.sh

# Build the Python extension into the active environment and run its tests.
python-test:
    cd tpt-rag-py && maturin develop --release
    pytest -q tpt-rag-py/tests

# Ingest the test PDFs and text fixtures and run a query with the Rust example. Needs `just setup`.
demo:
    RAGLITE_MODEL=models/all-MiniLM-L6-v2.onnx cargo run -p tpt-rag-core --example rust_basic -- ./demo_db tpt-rag-core/tests/fixtures "which planet is the red planet"

# Regenerate the C header and fail if it differs from the committed copy.
header:
    cbindgen --config tpt-rag-ffi/cbindgen.toml --crate tpt-rag-ffi --output tpt_raglite.h
    git diff --exit-code -- tpt_raglite.h
