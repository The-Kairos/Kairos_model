# Parallelization Benchmark Plan

This report compares Kairos production runs in two modes only:

- `semi_parallel`: current production orchestration with internal parallelism in audio and LLM stages.
- `parallel`: updated branch-parallel orchestration after scene detection.

## What We Track
- Total wall time per run
- Stage-level wall time from `checkpoint.json -> steps`
- Embedding provider and model
- Debug / quiet mode state
- Low-memory mode state

## Current Scope
- No sequential baseline
- No artificial `time.sleep(...)` delays in production processing
- Language detection logic remains unchanged
- Benchmark output is appended to `PARALLELIZATION_BENCHMARKS.md`

## Interpretation Notes
- In `parallel` mode, step wall times can sum to more than total wall time because branches overlap.
- The most important comparison is total wall time and the longest critical-path stages.
