"""CLI argument parsing."""

from __future__ import annotations

import argparse

from kairos.core.redo import REDO_CHOICES


def parse_args():
    parser = argparse.ArgumentParser(description="Process videos or run RAG.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    process = subparsers.add_parser("process", help="Process videos")
    process.add_argument("--video", action="append", help="Blob name or path (repeatable)")
    process.add_argument("--all", action="store_true", help="Process all catalog videos")
    process.add_argument("--filter", choices=["short", "medium", "long", "extra"], help="Inclusive length filter")
    process.add_argument("--include-unknown", action="store_true", help="Include videos with unknown length when filtering")
    process.add_argument("--preset", choices=["default", "fast", "motion", "static"], default="default", help="Pipeline config preset")
    process.add_argument("--llm", choices=["gemini", "openai"], default=None, help="LLM backend (overrides LLM_BACKEND env var)")
    process.add_argument("--redo", nargs="+", action="append", choices=REDO_CHOICES, help="Redo a processing step (repeatable)")
    process.add_argument("--redo-only", nargs="*", choices=REDO_CHOICES, help="Redo only specified steps (no dependents)")

    rag = subparsers.add_parser("rag", help="Run RAG for a single video")
    rag.add_argument("--video", required=True, help="Blob name or path")
    rag.add_argument("--llm", choices=["gemini", "openai"], default=None, help="LLM backend (overrides LLM_BACKEND env var)")

    return parser.parse_args()
