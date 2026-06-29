#!/bin/bash
# QVHighlights Test Split Benchmark — Full Pipeline
# Extracts test videos from tarball, runs Kairos pipeline in batches,
# then merges all predictions and evaluates with official metrics.
#
# Usage:
#   bash test/benchmarks/run_qvh_test_benchmark.sh           # Run all steps
#   bash test/benchmarks/run_qvh_test_benchmark.sh extract    # Extract test videos only
#   bash test/benchmarks/run_qvh_test_benchmark.sh batch N    # Run batch N (0-15)
#   bash test/benchmarks/run_qvh_test_benchmark.sh merge      # Merge and evaluate

set -e
cd "$(dirname "$0")/../../.."

TARBALL="test/benchmarks/cache/qvhighlights/qvhilights_videos.tar.gz"
BATCH_SIZE=100
TOTAL_VIDEOS=1529
NUM_BATCHES=$(( (TOTAL_VIDEOS + BATCH_SIZE - 1) / BATCH_SIZE ))

step_extract() {
    echo "=== Extracting test videos from tarball ==="
    python test/benchmarks/results/qvhighlights/run_qvhighlights_benchmark.py \
        --split test --download-tarball
}

step_batch() {
    local batch_num=$1
    local offset=$((batch_num * BATCH_SIZE))
    echo "=== Running batch $batch_num (offset=$offset, size=$BATCH_SIZE) ==="
    python test/benchmarks/results/qvhighlights/run_qvhighlights_benchmark.py \
        --split test \
        --batch-size $BATCH_SIZE \
        --batch-offset $offset \
        --top-k 5 \
        --merge-adjacent \
        --merge-gap-sec 5.0 \
        --mr-only \
        --output-cache-name qvhighlights_test_outputs
}

step_merge() {
    echo "=== Merging all test predictions ==="
    local pred_files=$(ls -1 test/benchmarks/results/qvhighlights/qvhighlights_predictions_*_merged.jsonl 2>/dev/null | sort)
    if [ -z "$pred_files" ]; then
        echo "ERROR: No prediction files found"
        exit 1
    fi
    python test/benchmarks/results/qvhighlights/run_qvhighlights_benchmark.py \
        --split test --mr-only \
        --merge-results $pred_files
}

case "${1:-all}" in
    extract)
        step_extract
        ;;
    batch)
        if [ -z "$2" ]; then
            echo "Usage: $0 batch N  (N = 0 to $((NUM_BATCHES-1)))"
            exit 1
        fi
        step_batch $2
        ;;
    merge)
        step_merge
        ;;
    all)
        step_extract
        echo ""
        echo "=== Running all $NUM_BATCHES batches ==="
        for i in $(seq 0 $((NUM_BATCHES-1))); do
            step_batch $i
            echo ""
        done
        step_merge
        ;;
    *)
        echo "Usage: $0 {extract|batch N|merge|all}"
        exit 1
        ;;
esac
