# Local complete LLM workflow comparison

Host recorded at run time: **the-Kewl-Laptop**, Windows 11, Python 3.12.1.
Both layouts run on this same local machine and use the same configured Azure deployment. No VM is involved.
All saved clues are reused; every scene report, intermediate narrative, and final report is generated anew.

| Input | Scenes per run | Original total median | Cache-friendly total median | Difference |
| --- | ---: | ---: | ---: | ---: |
| speech | 22 | 34.10s | 38.93s | +14.2% |
| story | 34 | 48.64s | 41.20s | -15.3% |
| visual | 16 | 32.47s | 41.67s | +28.3% |

Positive differences mean slower. Medians of stages do not necessarily sum to the median total.

## Stage wall times

| Input | Stage | Original median | Cache-friendly median |
| --- | --- | ---: | ---: |
| speech | scene_descriptions | 24.01s | 22.22s |
| speech | narrative | 5.49s | 5.24s |
| speech | synopsis | 4.94s | 12.71s |
| story | scene_descriptions | 28.82s | 27.80s |
| story | narrative | 7.40s | 8.12s |
| story | synopsis | 12.77s | 8.25s |
| visual | scene_descriptions | 10.68s | 23.59s |
| visual | narrative | 7.77s | 7.32s |
| visual | synopsis | 12.33s | 12.13s |

## Actual cache reads

| Input | Stage | Original cached input | Cache-friendly cached input |
| --- | --- | ---: | ---: |
| speech | scene_descriptions | 0.0% | 0.0% |
| speech | narrative | 0.0% | 0.0% |
| speech | synopsis | 0.0% | 58.6% |
| story | scene_descriptions | 3.6% | 2.5% |
| story | narrative | 0.0% | 0.0% |
| story | synopsis | 0.0% | 39.3% |
| visual | scene_descriptions | 0.0% | 0.0% |
| visual | narrative | 0.0% | 0.0% |
| visual | synopsis | 0.0% | 24.3% |

The short-scene and map/reduce narrative prompts have no experimental rearrangement; their timings can change because upstream generated inputs change.
The experimental layout rearranges full-scene instructions and shared-narrative synopsis prompts. A cache read is not proof of a faster complete stage.

## First trials and repeat trials

| Input | Layout | First trial | Later trial median |
| --- | --- | ---: | ---: |
| speech | original | 43.20s | 31.21s |
| speech | cache_friendly | 38.93s | 38.87s |
| story | original | 48.64s | 45.31s |
| story | cache_friendly | 52.67s | 40.93s |
| visual | original | 41.45s | 26.33s |
| visual | cache_friendly | 46.29s | 32.14s |

Median paired total-time ratio: 1.117. Exploratory bootstrap interval: [0.901, 1.283]. A ratio below 1 means faster.
Small sequential sample with shared service/network conditions; this does not prove equivalence.

## Output checks

| Run | Scenes kept | Questions | Fully missing required answers | Meaningful highlights/timeline/clips | Time fields not exactly in saved evidence |
| --- | --- | ---: | ---: | --- | ---: |
| speech_original_1 | True | 37 | 0 | 6/6/5 | 0 |
| speech_cache_friendly_1 | True | 37 | 0 | 5/6/5 | 0 |
| story_original_1 | True | 37 | 0 | 6/6/5 | 0 |
| story_cache_friendly_1 | True | 37 | 0 | 6/6/6 | 0 |
| visual_original_1 | True | 37 | 3 | 6/6/5 | 0 |
| visual_cache_friendly_1 | True | 37 | 4 | 5/6/5 | 0 |
| speech_cache_friendly_2 | True | 37 | 0 | 5/6/5 | 0 |
| speech_original_2 | True | 37 | 0 | 6/6/5 | 0 |
| story_cache_friendly_2 | True | 37 | 0 | 6/6/5 | 0 |
| story_original_2 | True | 37 | 0 | 6/6/5 | 0 |
| visual_cache_friendly_2 | True | 37 | 5 | 5/6/5 | 0 |
| visual_original_2 | True | 37 | 4 | 5/6/5 | 0 |
| speech_original_3 | True | 37 | 0 | 6/6/5 | 0 |
| speech_cache_friendly_3 | True | 37 | 0 | 6/6/5 | 0 |
| story_original_3 | True | 37 | 0 | 6/6/6 | 0 |
| story_cache_friendly_3 | True | 37 | 0 | 6/6/5 | 0 |
| visual_original_3 | True | 37 | 5 | 6/6/5 | 0 |
| visual_cache_friendly_3 | True | 37 | 4 | 5/6/5 | 0 |

API requests: 1026; failed attempts: 0; application retry attempts: 0.
Tree-reduce calls: 0. The configured production threshold is 50000 characters; reduction is not artificially forced.
Timestamp membership checks do not prove event alignment. Missing answers may be justified when evidence is absent.
Source-based factual spot review is recorded separately; structure alone does not prove equal quality.

## Limits

- Azure model inference and network latency are included in locally measured wall times.
- First trials are not verified cold; no cache clearing or artificial cache-busting is used.
- Repeats regenerate intermediate context; they are not necessarily exact-prompt cache repeats.
- Tree reduction runs only if the production 50000-character narrative threshold is exceeded.
- SDK retries are disabled for visibility; normal application retries remain enabled.
- Format and coverage checks are not exhaustive factual verification.
- First-trial connection startup is not isolated; compare first and later trials separately.
- No local transcription, detector, caption model, video decode, or embedding work is timed.
- Raw outputs and request traces stay local. Host, package versions, file/input hashes, and full settings are recorded in metrics.json.

Published summary only. Raw outputs, review bundles, and request traces referenced above remain in the local ignored run directory.
