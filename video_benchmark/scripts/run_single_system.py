"""
Thin wrapper: same as run_benchmark.py --systems <one_system>
Example: python scripts/run_single_system.py google_gemini
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_benchmark import main

if __name__ == "__main__":
    if len(sys.argv) >= 2 and not sys.argv[1].startswith("-"):
        name = sys.argv[1]
        sys.argv = [sys.argv[0], "--systems", name, *sys.argv[2:]]
    main()
