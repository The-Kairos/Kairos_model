#!/usr/bin/env python3
"""Convenience launcher for light VLM benchmarks (delegates to main_test.py).

Run from project root:
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
