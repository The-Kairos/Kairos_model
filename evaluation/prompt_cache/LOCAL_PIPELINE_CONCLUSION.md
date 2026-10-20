# Local full-workflow result

Executed on **the-Kewl-Laptop**, Windows 11, using the configured Azure
`gpt-4o-2024-11-20` model. Host and software metadata were recorded at run time.
No VM pipeline was used.

All 72 scenes across three short videos were processed in each paired repetition:
short scene summaries, full scene reports with five-scene history, fresh
narrative generation, and a fresh final report. There were three repetitions
per layout: eighteen full LLM workflows and 1,026 successful API calls. No
failed attempts, retries, or synopsis repair calls occurred.

| Input | Original median total | Cache-friendly median total | Result |
| --- | ---: | ---: | --- |
| Speech, 22 scenes | 34.10 s | 38.93 s | 14.2% slower |
| Story, 34 scenes | 48.64 s | 41.20 s | 15.3% faster |
| Visual, 16 scenes | 32.47 s | 41.67 s | 28.3% slower |

Scene descriptions generally consumed the largest stage wall time. Narrative
generation medians were approximately five to eight seconds. Final-report
medians ranged from roughly five to thirteen seconds.

Caching was concentrated in final-report input: 58.6% for the speech, 39.3% for
the story, and 24.3% for the visual clip under the experimental layout. Narrative
generation had zero reported cached input; scene-description reuse was zero or
small. Its short fixed prefixes generally did not yield useful cache reads.

The experimental ordering does **not** demonstrate a consistent speedup or
speed equivalence. The median paired total-time ratio was 1.117, with a broad
exploratory bootstrap interval of 0.901–1.283. The small sequential sample and
variable network/service conditions prevent a strong causal attribution to
caching alone.

The label-hidden local quality spot review found factual or temporal issues in
both layouts, including a wrong ending in one original report and unsupported
character attribution in one experimental report. Equal factual quality is not
established. See [QUALITY_REVIEW.md](LOCAL_PIPELINE_QUALITY.md).

**Recommendation:** keep the original prompt order as the default and leave
`KAIROS_CACHE_FRIENDLY_PROMPTS` disabled. Provider-side caching remains available;
the experimental reordering has not met the goal of consistently equal-or-better
speed with equivalent quality.

Scope limits: no ASR, caption/detection inference, video decode, or embedding work
was timed. The short inputs did not exceed the normal 50,000-character narrative
threshold, so live tree reduction was not exercised; a separate offline test
covers that branch. No whole-video or VM speed claim is made.

Detailed stage times, cache usage, first/repeat trials, and output checks are in
[RESULTS.md](LOCAL_PIPELINE_RESULTS.md); complete local metadata and aggregates are in
[metrics.json](LOCAL_PIPELINE_METRICS.json). Raw outputs remain local. The focused suite has
34 passing tests, including fresh-stage execution and saved-track text parity.

Published summary only. Raw outputs, review bundles, and request traces referenced above remain in the local ignored run directory.

Exact comparison: 0 of 9 paired final reports were identical. Different wording alone is not a quality failure; the separate source review found substantive differences as well.
