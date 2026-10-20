# Preserve synopsis answers and measure prompt caching safely

Video synopsis generation discarded valid answers when model-returned questions
omitted punctuation and could emit timestamps absent from the source narrative.
Normalize question matching and reject unsupported times while retaining
source-supported event descriptions. Clarify field separators and missing-time
instructions without changing the output schema.

Add opt-in request/stage telemetry and reproducible saved-input benchmarks.
Keep the original prompt order as the default: broader comparisons found the
shared-narrative rearrangement could be slower. That arrangement remains behind
`KAIROS_CACHE_FRIENDLY_PROMPTS=1` for further experiments; existing provider-side
caching remains available.

Validation: 28 focused tests cover both layouts, byte-identical default synopsis
prompts against the quality-fixed baseline, identical outputs for the same model
responses, answer retention, timestamp grounding, repair paths, file output,
scene-history ordering, concurrent metrics, and API error propagation.
The broader three-video comparison made 220 successful API calls with no failed
requests. See `evaluation/prompt_cache/MERGE_READINESS.md` for timings, release
smoke results, and limitations. No full-video speedup or exhaustive factual
equivalence is claimed.

Timing scope: these are fresh client-side `time.perf_counter()` measurements
around Azure requests launched from this workspace, not processing durations
copied from the saved input logs. They include network and Azure service latency;
they are not isolated local CPU timings or server-only inference timings.
The saved Titanic log identifies a different
machine (`Manahil-Laptop`); its historic duration was not used. The current
execution host identifies as `the-Kewl-Laptop`, but benchmark-host metadata was
not recorded at run time, so that identity cannot be independently established
from the original benchmark artifacts. These results do not establish VM
performance; no VM comparison or whole-video speedup has been demonstrated.

| Saved input | Original synopsis median | Rearranged synopsis median |
| --- | ---: | ---: |
| Titanic | 5.08 s | 7.04 s |
| Malala speech | 3.80 s | 5.76 s |
| CCTV | 4.80 s | 11.61 s |

Keep this PR in draft for review of the local performance and quality findings.
The experimental ordering remains off by default; no VM benchmark is required
for this local study.

### Follow-up: complete local LLM workflow

Recorded host at execution: `the-Kewl-Laptop` (Windows 11). No VM pipeline was
used. All scene reports, intermediate narratives, and final reports were
regenerated from saved non-LLM clues for three short videos, with three paired
repetitions: 18 workflows and 1,026 successful API calls, no failed attempts or
retries. Median total LLM times were 34.10 vs 38.93 seconds for speech, 48.64 vs
41.20 seconds for story, and 32.47 vs 41.67 seconds for visual footage
(original vs cache-friendly). Speed improvements were not consistent.

The outputs are not identical. Source-based spot review found differences in
selected content and factual/event-time details in both layouts; equal quality
is not established. Keep the experimental ordering off. See
`LOCAL_PIPELINE_CONCLUSION.md`, `LOCAL_PIPELINE_RESULTS.md`, and
`LOCAL_PIPELINE_QUALITY.md` in `evaluation/prompt_cache/`. Raw outputs and request
traces remain local. The follow-up focused suite passes 34 tests.
