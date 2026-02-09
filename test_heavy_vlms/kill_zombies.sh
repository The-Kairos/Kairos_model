#!/bin/bash
echo "=== Killing VLM Zombie Processes ==="
# Kill specific script processes
pkill -f "run_single_vlm.py"
pkill -f "process_base.py"
pkill -f "test_internvl.py"
pkill -f "test_qwenvl.py"
pkill -f "test_llava"

# Kill multiprocessing resource trackers
pkill -f "resource_tracker"

echo "=== Cleanup Complete ==="
