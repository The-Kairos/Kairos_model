#!/bin/bash

# Ensure we are in the script's directory FIRST
cd "$(dirname "$0")"

# Load environment variables from .env file
# First check parent directory (main Kairos_model folder)
if [ -f "../.env" ]; then
    set -a
    source ../.env
    set +a
    echo "✅ Loaded HF_TOKEN from ../env"
elif [ -f ".env" ]; then
    # Fallback: check current directory
    set -a
    source .env
    set +a
    echo "✅ Loaded HF_TOKEN from .env"
else
    echo "⚠️  No .env file found"
    echo "⚠️  Checked: $(cd .. && pwd)/.env"
    echo "⚠️  Checked: $(pwd)/.env"
    echo "⚠️  Running without authentication"
fi

VIDEOS_DIR="../Videos"
VLMS=("llava" "llava_mistral" "phi3v" "instructblip")

usage() {
    echo "Usage: $0 [base | vlm <vlm_name> | all]"
    echo "  base          - Run ASR/AST/YOLO for all videos (skips if exists)"
    echo "  vlm <name>    - Run specific VLM for all videos (skips if exists)"
    echo "  all           - Run full pipeline (base then all VLMs)"
    echo ""
    echo "Available VLMs: llava, llava_mistral, phi3v, instructblip"
    echo ""
    echo "Examples:"
    echo "  ./run_pipeline.sh base"
    echo "  ./run_pipeline.sh vlm llava"
    echo "  ./run_pipeline.sh vlm llava_mistral"
    echo "  ./run_pipeline.sh all"
    exit 1
}

run_base() {
    echo "===================================================="
    echo "STAGE 1: Base Processing (Extracting Features)"
    echo "===================================================="
    for vid in "$VIDEOS_DIR"/*.{mp4,avi,mov,mkv}; do
        [ -e "$vid" ] || continue
        filename=$(basename "$vid")
        name="${filename%.*}"
        python3 process_base.py "$vid" "results/base/$name/base_data.json"
    done
}

run_vlm() {
    VLM=$1
    echo "===================================================="
    echo "STAGE 2: VLM Inference ($VLM)"
    echo "===================================================="
    for vid in "$VIDEOS_DIR"/*.{mp4,avi,mov,mkv}; do
        [ -e "$vid" ] || continue
        filename=$(basename "$vid")
        name="${filename%.*}"
        python3 run_single_vlm.py "$VLM" "$vid" "results/base/$name/base_data.json" "results/$VLM/$name"
    done
}

case "$1" in
    base)
        run_base
        ;;
    vlm)
        if [ -z "$2" ]; then usage; fi
        run_vlm "$2"
        ;;
    all)
        run_base
        for m in "${VLMS[@]}"; do
            run_vlm "$m"
        done
        ;;
    *)
        usage
        ;;
esac
