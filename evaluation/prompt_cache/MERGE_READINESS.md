# Merge readiness

Prepare default-layout quality fixes and telemetry for merge; experimental rearrangement stays disabled.

## Performance decision

The experimental shared-narrative ordering did not satisfy the goal of equal or better speed.
Both layouts received the same answer-matching, separator, and timestamp safeguards in these comparisons.
Runs alternated order. These are saved-input LLM stages, not whole-video processing.

| Input | Runs per layout | Original synopsis median | Experimental synopsis median | Original scene sample median | Experimental scene sample median |
| --- | ---: | ---: | ---: | ---: | ---: |
| titanic | 5 | 5.08s | 7.04s | 3.85s | 2.66s |
| malala | 3 | 3.80s | 5.76s | 3.51s | 2.15s |
| cctv | 3 | 4.80s | 11.61s | 2.63s | 5.30s |

Total comparison requests: 220; failures: 0.
Scene timings were mixed; do not extrapolate three-scene samples to a complete video pipeline.
Cold-cache equivalence was not established. Repeated original prompts also cache automatically.

## Merge defaults

- Original task-first prompt order is the default. The runtime layout helper returns the prompt unchanged.
- Provider-side caching remains available with the existing API settings; telemetry reports actual cache usage.
- Experimental rearrangement requires `KAIROS_CACHE_FRIENDLY_PROMPTS=1`; it is not enabled by default.
- All seven primary synopsis prompts are byte-identical to the quality-fixed baseline in the integration fixture.
- The same mocked model responses produce identical complete synopsis objects for both default paths.
- No model, generation setting, concurrency, or normal retry policy changes were introduced.

## Quality and release smoke

Question matching tolerates punctuation differences without assigning answers by position.
The timestamp guard rejects unsupported values, preserves explicitly timed scene headings, and retains event descriptions.
Missing event times remain unknown; the guard does not recover lost timing or prove semantic alignment.

Final default smoke: 13.07s, 7 successful calls, 37 questions, 0 missing-answer placeholders.
Meaningful event descriptions: 6 highlights, 6 timeline events, 6 clips.
This single smoke run is a correctness check, not evidence of a speedup.

Local review found the sampled main summaries and required answers consistent with their saved narratives.
The review also exposed blank highlight descriptions in one baseline run; final prompts now explicitly preserve events when timing is missing.
The external judge was not run after automatic approval review rejected the additional data export.
There is no claim of full factual equivalence for every stochastic output.

## Validation

28 focused tests pass: prompt rendering and both layouts, answer matching, timestamp grounding, concurrent logging, unknown usage, API-error propagation, repairs, file output, scene-history ordering, custom-template preservation, and baseline request/output parity, including all three scene template API payloads.
Use the focused suite documented in README.md; unrelated video-model and live-connection tests are not part of this change.
`git diff --check` passes. Changes are prepared locally; no push or merge has been performed.
