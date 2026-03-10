#!/bin/bash

# Kairos Audio Pipeline Runner
# This script wraps the complex Python commands into a simple interface.

# Defaults
WORKERS=2
PARALLEL="--parallel"
CPU="--cpu"
DEBUG="--debug"

usage() {
    echo "Usage: ./run_pipeline.sh [options]"
    echo ""
    echo "Options:"
    echo "  --all            Process all videos in Videos/"
    echo "  --video [PATH]   Process a specific video file"
    echo "  --workers [N]    Number of workers (default: 2)"
    echo "  --gpu            Enable GPU (default: CPU-only)"
    echo "  --no-parallel    Disable parallel chunking"
    echo "  --parallel       Explicitly enable parallel chunking (default)"
    echo "  --cpu            Explicitly use CPU (default)"
    echo "  --language [LG]  Force language (e.g. ar, en)"
    echo "  --api            Use Azure Whisper API (default: True)"
    echo "  --debug          Enable debug logging"
    echo ""
    exit 1
}

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --all) ALL_FLAG=1 ;;
        --video) VIDEO_PATH="$2"; shift ;;
        --workers) WORKERS=$2; shift ;;
        --gpu) CPU_FLAG=0 ;;
        --cpu) CPU_FLAG=1 ;;
        --no-parallel) PARALLEL_FLAG=0 ;;
        --parallel) PARALLEL_FLAG=1 ;;
        --language) LG_VAL="$2"; shift ;;
        --api) API_FLAG=1 ;;
        --debug) DEBUG_FLAG=1 ;;
        *) usage ;;
    esac
    shift
done
echo "DEBUG: ALL_FLAG=$ALL_FLAG, VIDEO_PATH='$VIDEO_PATH', WORKERS=$WORKERS"

if [[ -z "$ALL_FLAG" && -z "$VIDEO_PATH" ]]; then
    usage
fi

# Build arguments array for Python
ARGS=()
[[ "$ALL_FLAG" == 1 ]] && ARGS+=("--all")
if [[ -n "$VIDEO_PATH" ]]; then
    ARGS+=("--video" "$VIDEO_PATH")
fi
[[ "$PARALLEL_FLAG" != 0 ]] && ARGS+=("--parallel")
ARGS+=("--workers" "$WORKERS")
[[ "$CPU_FLAG" != 0 ]] && ARGS+=("--cpu")
[[ -n "$LG_VAL" ]] && ARGS+=("--language" "$LG_VAL")
[[ "$DEBUG_FLAG" != 0 ]] && ARGS+=("--debug")
[[ "$API_FLAG" == 1 ]] && ARGS+=("--use-api")

# Run from project root
cd "$(dirname "$0")/.."

echo "🚀 Starting Kairos Audio Pipeline..."
PYTHONPATH=. python3 -m audio_singlecall.main "${ARGS[@]}"
