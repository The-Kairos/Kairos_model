# Titanic synopsis quality follow-up

Used saved narrative and captured outputs only; the video, transcription,
captioning, detection, and narrative generation were not rerun.

## Answer loss: identified and corrected

Captured API outputs contained answers but omitted punctuation from the required
question wording. The old question-key matching retained punctuation, so it
failed to associate the answers and substituted `Not explicitly stated.`.
Matching now normalizes punctuation while preserving word order. It does not
assign answers by position or match unrelated questions.

Offline replay of the same captured responses:

| Measurement | Old matching | Corrected matching |
| --- | ---: | ---: |
| Required answers lost to placeholders | 22/22 | 0/22 |

Run `python evaluation/titanic_sanity/run_summary_check.py` to reproduce the replay.
Machine-readable results: [metrics.json](metrics.json).

## Unsupported timestamps: rejected

The source final narrative retains only the overall video range in a heading,
not event-level timecodes. Overall ranges are excluded from timestamp evidence.
The final synopsis guard rejects timestamps absent from event-level source text,
normalizes equivalent timestamp precision for matching, and preserves descriptions.
The highlight, timeline, and clip prompts also explicitly forbid estimated times.

The captured output had 34 numeric timestamp fields, including 30 absent even
from the overall source range. All 34 become `Not explicitly stated` because
the source provides no event-aligned timecodes. No real timing is recovered by
this change. When a source does contain event timecodes, supported values remain.
Membership alone cannot prove a timecode was attached to the correct event.

## Fresh live check

- Azure returned model: `gpt-4o-2024-11-20`.
- Synopsis wall time: **10.04 seconds**.
- Successful API calls: **8**, including one timeline-format repair; no failed calls.
- Questions: **37**, all 22 required questions in the correct order.
- Missing-answer placeholders: **0**.
- Cached input: **92.6%**. This is a warmed follow-up, not a new speed comparison.
- Event timecodes remain unknown because the source lacks that information.

Artifacts: [live metrics](../prompt_cache/quality_fixed_20261020/metrics.json),
[final synopsis](../prompt_cache/quality_fixed_20261020/revised_1/synopsis.json),
[raw section responses](../prompt_cache/quality_fixed_20261020/revised_1/raw_sections.json).

Five new quality tests pass; the fourteen existing prompt/telemetry/benchmark
checks also pass. Answer retention and format checks do not establish full
factual accuracy. No whole-video speedup is claimed.

For useful event timestamps, the next separate improvement is to preserve
scene-aligned timecodes through narrative condensation rather than inventing
them during synopsis generation.
