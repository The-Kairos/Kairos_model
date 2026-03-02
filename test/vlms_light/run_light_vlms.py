#!/usr/bin/env python3
"""
Run light VLM benchmarks (same as main_test.py). Convenience launcher from project root:
  python test_light_vlms/run_light_vlms.py
"""
import runpy
import sys
from pathlib import Path

# Ensure project root is on path
root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

runpy.run_path(Path(__file__).parent / "main_test.py", run_name="__main__")
