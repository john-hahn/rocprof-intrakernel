#!/bin/bash
# run_all_examples.sh — Build and run all rocprof-intrakernel examples.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${ROOT_DIR}/build"

echo "=== Building rocprof-intrakernel ==="
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"
cmake "$ROOT_DIR" -DRIKP_BUILD_EXAMPLES=ON -DRIKP_BUILD_TESTS=ON
make -j"$(nproc)"

echo ""
echo "=== Running trace examples ==="

echo "--- record ---"
./rikp_trace_record --iters=1000 --out=record_trace.json

echo "--- block_filter ---"
./rikp_trace_block_filter --iters=1000 --out=block_filter_trace.json

echo "--- sampled_loop ---"
./rikp_trace_sampled_loop --iters=100000 --sample_shift=10 --out=sampled_loop_trace.json

echo "--- gemm ---"
./rikp_gemm_demo --m=512 --n=512 --k=512 --out=gemm_trace.json

echo ""
echo "=== Running tests ==="
ctest --output-on-failure

echo ""
echo "=== Generating explorer ==="
python3 "$SCRIPT_DIR/generate_explorer.py" \
    --trace gemm_trace.json \
    --summary gemm_trace_summary.json \
    -o gemm_explorer.html

echo ""
echo "=== All done ==="
echo "Trace files: record_trace.json, block_filter_trace.json, sampled_loop_trace.json, gemm_trace.json"
echo "Explorer: gemm_explorer.html"
