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

Keep this PR in draft pending a comparison on the target VM and further
source-based quality review. The experimental ordering remains off by default.
