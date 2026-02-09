#!/bin/bash
echo "=== Killing VLM Zombie Processes ==="
# Kill orchestrator and wrappers
pkill -f "run_benchmark.py"
pkill -f "run_single_vlm.py"
pkill -f "process_base.py"

# Kill specific VLM processes
pkill -f "test_llava_1_6"
pkill -f "test_phi3v"
pkill -f "test_llama32"
pkill -f "test_internvl"
pkill -f "test_qwenvl"

# Kill multiprocessing resource trackers
pkill -f "resource_tracker"
pkill -f "multiprocessing"

echo "=== Cleanup Complete ==="
