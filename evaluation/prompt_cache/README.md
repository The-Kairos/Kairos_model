# Prompt caching benchmark

Start with [MERGE_READINESS.md](MERGE_READINESS.md) for the current merge decision.
The default prompt order is restored. Experimental rearrangement is disabled
unless `KAIROS_CACHE_FRIENDLY_PROMPTS=1` is set. Provider-side caching is not
disabled. The earlier six-run report below is historical.

The live Titanic comparison is in [titanic_20261020_run2/RESULTS.md](titanic_20261020_run2/RESULTS.md).
This run used Azure's returned model `gpt-4o-2024-11-20`, three repetitions per
layout, identical saved inputs, and unchanged generation settings/concurrency.

To repeat with a new output directory:

```powershell
python evaluation/prompt_cache/run_benchmark.py --baseline-ref c32882928a4650f77717a9379acd9593a42c45bb --output evaluation/prompt_cache/new_run --repetitions 3 --scene-samples 3 --match-quality-fixes --experimental-layout
python evaluation/prompt_cache/analyze_results.py evaluation/prompt_cache/new_run
```

The baseline code and historical templates are read from Git without modifying
the working tree. With `--match-quality-fixes`, both arms use current task-first
templates and identical quality safeguards; only the candidate loader applies
the optional rearrangement. The working-tree pipeline supplies the revised arm. No API cache key,
retention parameter, warm-up request, or model change is added. A two-token
connectivity probe runs first. SDK retries are disabled in the benchmark so API
attempts are visible; the existing application retry loops remain enabled.

Request telemetry is opt-in in normal pipeline runs:

```powershell
$env:KAIROS_LLM_METRICS_PATH = 'evaluation/prompt_cache/pipeline_requests.jsonl'
$env:KAIROS_LLM_RUN_ID = 'my_pipeline_run'
```

Records include stage, model/deployment, timing, usage, cached input tokens,
outer attempt number, finish reason, and error class/status. An absent cache
usage field stays null rather than being treated as a miss. Log writes are
thread-safe within one process. Use a separate file for each process. SDK-internal
retries are included in one record's elapsed time and are not separately counted.
Scene Gemini calls are not instrumented by the Azure chat-completion wrapper.
Complete scene/narrative/synopsis stage wall times include parallel work.

Request telemetry excludes prompt/response text, endpoint URLs, and credentials.
Benchmark artifacts separately retain generated synopsis and scene outputs for
review. New runs also retain `raw_sections.json` to distinguish model responses
from parser/normalization failures. The original six-run comparison predates
this capture and does not include those intermediate responses.

The quality follow-up is documented in
[../titanic_sanity/RESULTS.md](../titanic_sanity/RESULTS.md). One-arm diagnostic
runs can use `--variant revised --repetitions 1 --scene-samples 0`; the comparison
analyzer expects both arms and is not needed for such a run.

The standalone scene sample serializes saved evidence to JSON; it does not
reproduce production two-stage scene history. Whole-pipeline timing, different
videos, and fresh unrelated narratives remain outside this small experiment.

Verification:

```powershell
python -m unittest discover -s test -p test_llm_metrics.py -v
python -m unittest discover -s test -p test_prompt_layout.py -v
python -m unittest discover -s test -p test_cache_benchmark.py -v
python -m unittest discover -s test -p test_synopsis_quality.py -v
python -m unittest discover -s test -p test_pipeline_regressions.py -v
```
