# Prompt layout and merge defaults

The original task-first layout is the default. Provider-side prompt caching is
still available with the existing API settings. The rearrangements described
below require `KAIROS_CACHE_FRIENDLY_PROMPTS=1` and are experimental: broader live
comparisons found latency regressions. See
[merge readiness](../evaluation/prompt_cache/MERGE_READINESS.md).

The source templates keep their original order. `src/prompt_layout.py` applies
the following transformations only when explicitly enabled:

Fixed instructions and output schemas precede changing scene evidence and
carryover context. Scene templates keep the video name ahead of scene evidence
so requests for the same video can also share that portion of the prefix.

Synopsis tasks reuse the full narrative before their different task instructions.
The eight narrative templates and the inline fallback, question-fill, legacy
question, and consistency builders use the same prefix. Keep their prefix text
and whitespace identical; `narrative_prefix` in `src/prompt_layout.py` is the
reference for the inline builders. Output schemas remain the same. Separately,
question matching and timestamp grounding have explicit quality fixes and tests.

The following prompts were reviewed and left in their existing order because
their fixed instructions already precede changing input:

- `describe_scene_short.txt`
- `chunk_summary_carryover.txt`
- `generate_answer.txt` (Gemini retrieval answers)
- Inline map, reduce, narrative consistency, and JSON repair prompts
- The final fallback that returns draft JSON unchanged

No filler, cache warm-up calls, response caching, model changes, or new API
parameters are introduced. Prefix layout enables potential provider-side reuse;
it does not guarantee a hit. Short instruction prefixes may still be below the
deployment's cache threshold. Concurrent first-use synopsis calls may all miss.

Before claiming a speed improvement, measure Azure deployment cache support,
`usage.prompt_tokens_details.cached_tokens`, request and stage wall times,
retries, and output quality on the same inputs. The latest Titanic sanity
results were unavailable when these edits were made. A subsequent live saved-input
comparison is available in
[`evaluation/prompt_cache/titanic_20261020_run2/RESULTS.md`](../evaluation/prompt_cache/titanic_20261020_run2/RESULTS.md).
It confirms synopsis cache reuse but does not establish a speed improvement.

Offline verification:

```powershell
python -m unittest discover -s test -p test_prompt_layout.py -v
```

Reference: [OpenAI prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching).
