# Benchmarking Publication Note

## Current State

- Kairos benchmarking is already implemented for two tasks:
- `SceneWalk`: scene-level description evaluation with `SODA`, `BERTScore`, and `ROUGE-L`
- `TIB`: full-video synopsis evaluation with `BERTScore`, `ROUGE-L`, and `BLEU`
- Current checked-in reports:
- `test/benchmarks/results/scenewalk_benchmark_report.md`
- `test/benchmarks/results/tib_benchmark_report.md`

## What SODA Is Used For

- `SODA` is already used in the SceneWalk benchmark.
- It evaluates temporal alignment between Kairos scene descriptions and SceneWalk reference segments.
- In this repo, the benchmark runner imports and computes `SODA` in `test/benchmarks/run_scenewalk_benchmark.py`.

## Methodology Risk

- Prompt tuning is not inherently cheating.
- It becomes methodologically weak if we tune directly on the same benchmark videos that we later present as final held-out test results.
- For a publishable evaluation, the correct protocol is:
- use a small development subset for prompt iteration
- freeze the final prompt
- run final evaluation on disjoint held-out videos

## Publishable Evaluation Plan

1. Treat the current 2-video SceneWalk run as development-only prompt-ablation evidence.
2. Freeze one final version of the scene-description prompts.
3. Re-run SceneWalk on additional unseen benchmark videos for final reported numbers.
4. Keep TIB as a separate whole-video synopsis task and report it independently from SceneWalk.
5. Disclose prompt tuning explicitly in the paper as part of the multimodal fusion design.
6. Emphasize `BERTScore` as the primary semantic metric and `SODA` as the temporal alignment metric.
7. Present `ROUGE-L` and `BLEU` as secondary lexical-overlap metrics.

## Why Prompt Tuning Is Defensible

- Kairos does not output labels directly from BLIP, YOLO, ASR, and AST.
- It uses an LLM fusion stage to transform multimodal evidence into scene descriptions.
- The wording and constraints of that fusion prompt are therefore part of the model pipeline, not a post-hoc formatting trick.
- Prompt ablations can be reported as a system-design choice, provided the final prompt is frozen before held-out evaluation.

## What We Are Tuning

- The main tuning target is the scene-description fusion stage:
- `prompts/describe_scene.txt`
- `prompts/fallback_describe_scene.txt`
- The short contextual summarization stage also influences the final report:
- `prompts/describe_scene_short.txt`

## Observed SceneWalk Failure Modes

- Kairos descriptions are often longer than SceneWalk references.
- Kairos sometimes adds fine-grained visual attributes that are not strongly supported.
- Kairos sometimes sounds cinematic or interpretive rather than directly visual.
- SceneWalk references are usually more compact, more literal, and more event-focused.
- Overly specific clothing, emotion, or setting details can hurt semantic alignment when unsupported.

## First Prompt-Tuning Direction

- Make outputs more grounded and compact.
- Lead with directly visible actions, actors, and setting.
- Use audio only to support visible events, not to speculate beyond them.
- Avoid cinematic filler, vague interpretation, and decorative detail.
- Prefer one coherent scene account over a long flourish.

## Paper Wording Draft

### Benchmarking Section

Kairos generates scene descriptions through a multimodal fusion stage that combines BLIP captions, YOLO detections, ASR transcripts, and AST audio cues into a single LLM-mediated scene report. Because benchmark evaluation is text-based, prompt formulation materially affects how faithfully the fused representation is rendered into natural language. We therefore performed prompt ablations on a small development subset and selected a single final prompt before held-out evaluation.

For SceneWalk, we evaluate scene-level descriptions using `SODA` to measure temporal alignment and `BERTScore` to measure semantic similarity between matched descriptions. We also report `ROUGE-L` as a secondary lexical-overlap metric. For TIB, we compare Kairos full-video synopses against human-written abstracts using `BERTScore`, `ROUGE-L`, and `BLEU`.

We treat `BERTScore` as the primary semantic metric because Kairos often produces valid paraphrastic descriptions whose meaning is preserved even when exact wording differs from the reference. In contrast, `ROUGE-L` and `BLEU` are more sensitive to phrasing and therefore under-reward semantically correct but differently worded outputs.

### Prompt-Tuning Disclosure

Prompt tuning is part of the Kairos scene-fusion pipeline rather than a post-processing step. However, to avoid benchmark-specific overfitting, all prompt iterations were conducted on a development subset, after which the final prompt was frozen and evaluated on disjoint held-out videos.

## Development Prompt Ablation Record

The following SceneWalk results are development-only results from the same 2-video prompt-tuning subset. They should not be reported as final held-out benchmark evidence. Their purpose is to justify prompt selection and document the engineering decisions made before freezing a final prompt.

### Pre-Versioning Baseline

- version label: `pre_versioning_dev_baseline`
- purpose:
- initial reference before formal prompt version tracking
- prompt style:
- active scene-description prompts before the formal version ledger
- development results:
- `Matched BERTScore F1`: `0.5723`
- `BERTScore Precision`: `0.5212`
- `BERTScore Recall`: `0.6352`
- `Matched ROUGE-L F1`: `0.1771`
- `SODA F1`: `0.0686`
- `SODA Precision`: `0.0443`
- `SODA Recall`: `0.1523`
- interpretation:
- the baseline had relatively high BERTScore recall but lower precision, indicating that Kairos included broad semantic coverage but also introduced extra or mismatched details
- SODA and ROUGE-L were weak, suggesting temporal and lexical mismatch with SceneWalk references

### Prompt Candidate `v1`

- version label: `v1`
- prompt files:
- `prompts/benchmark_versions/describe_scene_v1.txt`
- `prompts/benchmark_versions/fallback_describe_scene_v1.txt`
- `prompts/benchmark_versions/describe_scene_short_v1.txt`
- tuning goal:
- make descriptions more grounded, compact, and visual-first
- reduce unsupported specifics, cinematic filler, and decorative interpretation
- development results:
- `Matched BERTScore F1`: `0.5745`
- `Matched ROUGE-L F1`: `0.2208`
- `SODA F1`: `0.0859`
- `SODA Precision`: `0.0553`
- `SODA Recall`: `0.1925`
- `Total matched pairs`: `113`
- comparison to baseline:
- `Matched BERTScore F1`: `0.5723 -> 0.5745` (`+0.0022`)
- `Matched ROUGE-L F1`: `0.1771 -> 0.2208` (`+0.0437`)
- `SODA F1`: `0.0686 -> 0.0859` (`+0.0173`)
- interpretation:
- `v1` improved temporal/segment alignment and lexical overlap substantially
- BERTScore improved only slightly, but the prompt was a better balanced candidate than the baseline
- the main remaining problem was that outputs still sometimes included unsupported clothing, identity, location, emotion, or dialogue detail

### Prompt Candidate `v2`

- version label: `v2`
- prompt files:
- `prompts/benchmark_versions/describe_scene_v2.txt`
- `prompts/benchmark_versions/fallback_describe_scene_v2.txt`
- `prompts/benchmark_versions/describe_scene_short_v2.txt`
- tuning goal:
- make the prompt more conservative
- reduce unsupported attributes and transcript-driven details
- emphasize compact visual grounding over broad narrative coverage
- development results:
- `Matched BERTScore F1`: `0.5771`
- `BERTScore Precision`: `0.6015`
- `BERTScore Recall`: `0.5556`
- `Matched ROUGE-L F1`: `0.2095`
- `SODA F1`: `0.0815`
- `SODA Precision`: `0.0525`
- `SODA Recall`: `0.1827`
- `Total matched pairs`: `113`
- comparison to `v1`:
- `Matched BERTScore F1`: `0.5745 -> 0.5771` (`+0.0026`)
- `BERTScore Precision`: `0.5803 -> 0.6015` (`+0.0212`)
- `BERTScore Recall`: `0.5697 -> 0.5556` (`-0.0141`)
- `Matched ROUGE-L F1`: `0.2208 -> 0.2095` (`-0.0113`)
- `SODA F1`: `0.0859 -> 0.0815` (`-0.0044`)
- interpretation:
- `v2` improved the primary BERTScore F1 by raising precision
- the tradeoff was lower recall, SODA, and ROUGE-L because the descriptions became too conservative and lost some SceneWalk-like event coverage
- this showed that precision alone was not enough; the next prompt needed to recover recall without returning to hallucinated detail

### Prompt Candidate `v3`

- version label: `v3`
- prompt files:
- `prompts/benchmark_versions/describe_scene_v3.txt`
- `prompts/benchmark_versions/fallback_describe_scene_v3.txt`
- `prompts/benchmark_versions/describe_scene_short_v3.txt`
- tuning goal:
- preserve `v2`'s grounded visual style while restoring SceneWalk-like structure
- explicitly describe setting, main subject, main action, shot progression, and key objects
- improve BERTScore recall and overall F1 without benchmark wording mimicry
- development results:
- `Matched BERTScore F1`: `0.5836`
- `BERTScore Precision`: `0.5933`
- `BERTScore Recall`: `0.5750`
- `Matched ROUGE-L F1`: `0.2256`
- `SODA F1`: `0.0878`
- `SODA Precision`: `0.0566`
- `SODA Recall`: `0.1968`
- `Total matched pairs`: `113`
- comparison to `v2`:
- `Matched BERTScore F1`: `0.5771 -> 0.5836` (`+0.0065`)
- `BERTScore Precision`: `0.6015 -> 0.5933` (`-0.0082`)
- `BERTScore Recall`: `0.5556 -> 0.5750` (`+0.0194`)
- `Matched ROUGE-L F1`: `0.2095 -> 0.2256` (`+0.0161`)
- `SODA F1`: `0.0815 -> 0.0878` (`+0.0063`)
- interpretation:
- `v3` was the strongest development candidate so far by the primary BERTScore F1
- it also improved SODA and ROUGE-L over both `v1` and `v2`
- the improvement supports the idea that SceneWalk alignment requires not only compact grounding but also human-reference-like event structure
- the remaining risk was that some outputs reintroduced too much transcript/audio detail

### Prompt Candidate `v4`

- version label: `v4`
- prompt files:
- `prompts/benchmark_versions/describe_scene_v4.txt`
- `prompts/benchmark_versions/fallback_describe_scene_v4.txt`
- `prompts/benchmark_versions/describe_scene_short_v4.txt`
- tuning goal:
- keep `v3`'s SceneWalk-like visual structure
- tighten audio and dialogue use so transcript content does not dominate visible action
- reduce non-visible speech carryover without losing event coverage
- development results:
- `Matched BERTScore F1`: `0.5830`
- `BERTScore Precision`: `0.6006`
- `BERTScore Recall`: `0.5671`
- `Matched ROUGE-L F1`: `0.2284`
- `SODA F1`: `0.0886`
- `SODA Precision`: `0.0570`
- `SODA Recall`: `0.1986`
- `Total matched pairs`: `113`
- comparison to `v3`:
- `Matched BERTScore F1`: `0.5836 -> 0.5830` (`-0.0006`)
- `BERTScore Precision`: `0.5933 -> 0.6006` (`+0.0073`)
- `BERTScore Recall`: `0.5750 -> 0.5671` (`-0.0079`)
- `Matched ROUGE-L F1`: `0.2256 -> 0.2284` (`+0.0028`)
- `SODA F1`: `0.0878 -> 0.0886` (`+0.0008`)
- interpretation:
- `v4` improved SODA and ROUGE-L slightly and increased BERTScore precision
- however, its stricter audio restraint reduced recall enough that BERTScore F1 dropped slightly relative to `v3`
- if the final selection criterion prioritizes BERTScore F1, `v3` remains the strongest prompt candidate from the development set
- if the final selection criterion prioritizes slightly better temporal/lexical alignment and stricter visual grounding, `v4` is a defensible alternative

## Development Result Summary Table

| Version | Tuning Goal | BERTScore F1 | Precision | Recall | ROUGE-L F1 | SODA F1 |
|---|---|---:|---:|---:|---:|---:|
| Baseline | pre-versioning reference | `0.5723` | `0.5212` | `0.6352` | `0.1771` | `0.0686` |
| `v1` | compact grounded visual descriptions | `0.5745` | not recorded | not recorded | `0.2208` | `0.0859` |
| `v2` | stricter conservative grounding | `0.5771` | `0.6015` | `0.5556` | `0.2095` | `0.0815` |
| `v3` | SceneWalk-like event structure and recall | `0.5836` | `0.5933` | `0.5750` | `0.2256` | `0.0878` |
| `v4` | `v3` structure with stricter audio restraint | `0.5830` | `0.6006` | `0.5671` | `0.2284` | `0.0886` |

## Temporal Aggregation / Scene Merging Rationale

SceneWalk annotations and Kairos scene detections operate at different temporal granularities. In the 2-video development subset, Kairos produced many more scene units than the SceneWalk ground truth:

- Video `mDvkux01G3A`: Kairos detected `301` scenes, while SceneWalk provides `80` ground-truth segments.
- Video `X9MAf245Yag`: Kairos detected `166` scenes, while SceneWalk provides `51` ground-truth segments.

This mismatch means that a single SceneWalk reference often describes a broader event span than one Kairos scene description. A Kairos description can therefore be locally correct but still score poorly because it is matched against a reference that includes neighboring actions before or after the detected Kairos cut.

For this reason, temporal aggregation or scene merging is a defensible benchmarking step if it is applied before held-out evaluation using fixed, documented rules. The purpose is not to alter semantic content after seeing the references, but to align the evaluation unit with the annotation unit used by SceneWalk.

Acceptable aggregation rules should be:

- predefined before held-out evaluation
- applied identically to all held-out videos
- independent of the reference wording
- based only on Kairos timestamps, scene order, and optionally fixed time windows or adjacent-scene grouping
- reported clearly in the methodology section

Examples of defensible aggregation strategies:

- merge adjacent Kairos scenes until their combined duration approximates the median SceneWalk segment duration
- merge adjacent Kairos scenes within a fixed time window, such as 20-30 seconds
- merge very short Kairos scenes with neighboring scenes when they fall below a minimum duration threshold
- create a benchmark-only segment description by summarizing several adjacent Kairos scene descriptions without changing the underlying model pipeline

The key methodological distinction is that aggregation should align temporal units, not tune wording to match a particular reference. If aggregation is used, both unaggregated and aggregated development results can be reported internally, but only the frozen aggregation policy should be used for held-out results.

## Temporal Aggregation Development Results

After prompt tuning, fixed-window temporal aggregation was added as a separate development experiment. This did not remove or replace any prompt-version results. It evaluates whether Kairos descriptions should be scored at a slightly broader temporal unit to match SceneWalk's annotation style.

Implementation details:

- aggregation is implemented in `test/benchmarks/run_scenewalk_benchmark.py`
- CLI options:
- `--aggregate-predictions fixed_window`
- `--aggregation-window-sec <seconds>`
- `--aggregation-max-gap-sec <seconds>`
- the policy merges adjacent Kairos prediction segments by timestamp
- the policy uses only Kairos scene order, Kairos timestamps, a fixed window size, and a fixed max-gap threshold
- the policy does not use SceneWalk reference captions or semantic similarity during merging
- all original unaggregated prompt outputs remain preserved

Development search results:

| Prompt/output source | Aggregation policy | Window | BERTScore F1 | Precision | Recall | ROUGE-L F1 | SODA F1 | Matched pairs |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `v3` output | none | none | `0.5836` | `0.5933` | `0.5750` | `0.2256` | `0.0878` | `113` |
| `v4` output | none | none | `0.5830` | `0.6006` | `0.5671` | `0.2284` | `0.0886` | `113` |
| `v3` output | `fixed_window` | `10s` | `0.5861` | not recorded | not recorded | `0.2311` | `0.1244` | `126` |
| `v3` output | `fixed_window` | `13s` | `0.5851` | not recorded | not recorded | `0.2318` | `0.1429` | `129` |
| `v3` output | `fixed_window` | `15s` | `0.5841` | not recorded | not recorded | `0.2294` | `0.1528` | `129` |
| `v3` output | `fixed_window` | `20s` | `0.5835` | not recorded | not recorded | `0.2207` | `0.1722` | `129` |
| `v3` output | `fixed_window` | `25s` | `0.5825` | not recorded | not recorded | `0.2127` | `0.1792` | `128` |
| `v3` output | `fixed_window` | `30s` | `0.5800` | not recorded | not recorded | `0.2041` | `0.1840` | `124` |
| `v3` output | `fixed_window` | `45s` | `0.5784` | not recorded | not recorded | `0.1884` | `0.1521` | `91` |
| `v4` output | `fixed_window` | `8s` | `0.5841` | not recorded | not recorded | `0.2305` | `0.1065` | `121` |
| `v4` output | `fixed_window` | `10s` | `0.5862` | not recorded | not recorded | `0.2346` | `0.1258` | `126` |
| `v4` output | `fixed_window` | `12s` | `0.5878` | not recorded | not recorded | `0.2357` | `0.1397` | `129` |
| `v4` output | `fixed_window` | `13s` | `0.5879` | `0.5938` | `0.5834` | `0.2358` | `0.1450` | `129` |
| `v4` output | `fixed_window` | `14s` | `0.5873` | `0.5908` | `0.5850` | `0.2366` | `0.1509` | `130` |

Best development aggregation candidate:

- prompt/output source: `v4`
- aggregation policy: `fixed_window`
- window: `13s`
- max gap: `5s`
- `Matched BERTScore F1`: `0.5879`
- `BERTScore Precision`: `0.5938`
- `BERTScore Recall`: `0.5834`
- `Matched ROUGE-L F1`: `0.2358`
- `SODA F1`: `0.1450`
- `SODA Precision`: `0.1052`
- `SODA Recall`: `0.2342`
- `Total matched pairs`: `129`

Interpretation:

- fixed-window aggregation improves BERTScore F1 from the best unaggregated prompt result (`0.5836`) to `0.5879`
- fixed-window aggregation improves SODA much more strongly, from `0.0886` unaggregated `v4` to `0.1450` with 13-second aggregation
- wider windows improve temporal alignment but can reduce semantic precision because the merged descriptions become too broad
- short windows around `12-13s` provide the best BERTScore tradeoff on the development set
- if BERTScore remains the primary selection metric, the best current development configuration is `v4` with `fixed_window`, `13s` window, and `5s` max gap
- this policy must still be frozen before held-out evaluation and should not be retuned on held-out videos

## Paper Wording Draft For Temporal Aggregation

SceneWalk ground-truth annotations are written over temporally extended segments, whereas Kairos detects and describes a larger number of shorter scene units. This creates an evaluation granularity mismatch: a Kairos scene may accurately describe a local visual moment while the corresponding SceneWalk reference includes neighboring actions within the same annotated segment. To address this, we evaluate a fixed temporal aggregation variant in which adjacent Kairos scene descriptions are merged before scoring. The aggregation policy is selected on the development set and then frozen before held-out evaluation. It uses only Kairos timestamps and scene order, not reference text, and is applied uniformly across held-out videos.

We report this as an evaluation-alignment step rather than a change to the underlying multimodal evidence extraction pipeline. The underlying Kairos model still performs scene detection, frame captioning, object detection, audio analysis, and LLM-based fusion; temporal aggregation only adjusts the unit of comparison so it better matches the SceneWalk annotation granularity.

## Recommended Next Benchmarking Step

This section records the pre-aggregation recommendation that led to the fixed-window aggregation experiments below. The current recommendation after aggregation testing is listed in `Current Recommended Next Step After Aggregation`.

Based on the development prompt ablations, prompt wording alone improved BERTScore F1 from `0.5723` to `0.5836`, but did not reach the target range of `0.6` to `0.7`. The most likely ceiling is temporal granularity mismatch rather than only description phrasing.

The next controlled step should therefore be:

1. Select the best prompt candidate from the development ledger, currently `v3` by BERTScore F1.
2. Define one or two fixed temporal aggregation policies.
3. Evaluate those policies only on the development subset.
4. Choose one final prompt plus one final aggregation policy.
5. Freeze both before held-out SceneWalk evaluation.
6. Report held-out results separately from development prompt/aggregation ablations.

## Important Files To Review

- `prompts/describe_scene.txt`
- `prompts/fallback_describe_scene.txt`
- `prompts/describe_scene_short.txt`
- `src/scene_description.py`
- `test/benchmarks/run_scenewalk_benchmark.py`
- `test/benchmarks/metrics/soda_metric.py`
- `test/benchmarks/run_tib_benchmark.py`
- `test/benchmarks/results/scenewalk_comparison.md`
- `test/benchmarks/results/scenewalk_benchmark_report.md`
- `test/benchmarks/results/tib_benchmark_report.md`

## Current Recommended Next Step After Aggregation

The current best development configuration is:

- prompt/output source: `v4`
- aggregation policy: `fixed_window`
- window: `13s`
- max gap: `5s`
- development `Matched BERTScore F1`: `0.5879`
- development `SODA F1`: `0.1450`
- development `Matched ROUGE-L F1`: `0.2358`

This is better than the best unaggregated prompt result, but it still does not reach the target range of `0.6` to `0.7`. The next improvement should therefore not be another ordinary scene-description prompt pass. The next controlled experiment should be an abstractive aggregation layer.

### Why Fixed Concatenation Is Not Enough

The current `fixed_window` policy merges neighboring Kairos descriptions by concatenating them. This improves temporal alignment, but the merged text can become repetitive or overly broad. BERTScore may not rise enough because the prediction still reads like several scene descriptions stitched together rather than one human-written SceneWalk-style segment caption.

### Next Controlled Experiment: Abstractive Aggregation

The next controlled step should create a benchmark-only segment description after fixed-window merging:

1. Keep all existing Kairos scene descriptions unchanged.
2. Merge adjacent Kairos scene descriptions using the frozen candidate policy, currently `fixed_window`, `13s`, `5s max gap`.
3. Send each merged group to an LLM with a new aggregation prompt.
4. Ask the LLM to rewrite the grouped scene descriptions as one compact SceneWalk-style segment caption.
5. Save this as a new benchmark aggregation version, separate from the base scene prompt versions.
6. Evaluate on the same development subset.
7. If it improves BERTScore without obvious overfitting, freeze both:
- the base scene prompt
- the aggregation rewrite prompt and policy
8. Run held-out SceneWalk only after both are frozen.

This remains methodologically defensible if the aggregation rewrite uses only Kairos outputs and timestamps, not SceneWalk reference text.

### Expected Benefit

Abstractive aggregation may break the current `0.59` ceiling because it addresses both problems at once:

- temporal mismatch: multiple short Kairos scenes become one segment-level prediction
- textual mismatch: stitched descriptions become one coherent human-reference-like caption

This is the most plausible next path toward `0.6+` BERTScore without overfitting to SceneWalk wording.

## Next Actions

1. Preserve all existing prompt-version and fixed-window aggregation results.
2. Add a new aggregation-rewrite prompt under `prompts/benchmark_versions/`.
3. Add a benchmark-only abstractive aggregation mode to the SceneWalk runner.
4. Test it on the same 2-video development subset only.
5. Compare against the current best result: `v4 + fixed_window 13s`.
6. Freeze the best prompt plus aggregation policy before held-out evaluation.
7. Run held-out SceneWalk evaluation on additional unseen videos.
8. Expand TIB to a larger held-out sample.
9. Add one short error-analysis paragraph to the paper using examples from `scenewalk_comparison.md`.

## Staged Benchmarking Decision Log

This section records the development process in the order decisions were made. It should be treated as an internal audit trail for prompt and aggregation selection, not as final held-out benchmark evidence.

### Stage 1: Original Results

The original 2-video SceneWalk development run established the pre-versioning baseline:

| Configuration | BERTScore F1 | Precision | Recall | ROUGE-L F1 | SODA F1 | Matched pairs |
|---|---:|---:|---:|---:|---:|---:|
| Original baseline | `0.5723` | `0.5212` | `0.6352` | `0.1771` | `0.0686` | not recorded |

### Stage 2: Recommendation

The first recommendation was to tune the scene-description prompt carefully while preserving strict version history. The goal was to make Kairos outputs more comparable to SceneWalk captions by improving grounded visual structure, not by copying the reference wording.

Rules applied:

- save every tested prompt as a new versioned file
- log every result
- use the 2-video run only as a development set
- avoid held-out evaluation until the selected prompt is frozen

### Stage 3: Recommendation Tried - New Results

Prompt versions `v1` through `v4` were tested. Prompt tuning improved the original BERTScore F1 from `0.5723` to a best unaggregated score of `0.5836`.

| Configuration | What It Was Tuned For | BERTScore F1 | Precision | Recall | ROUGE-L F1 | SODA F1 |
|---|---|---:|---:|---:|---:|---:|
| Original baseline | pre-versioning reference | `0.5723` | `0.5212` | `0.6352` | `0.1771` | `0.0686` |
| `v1` | compact grounded visual descriptions | `0.5745` | not recorded | not recorded | `0.2208` | `0.0859` |
| `v2` | stricter conservative grounding | `0.5771` | `0.6015` | `0.5556` | `0.2095` | `0.0815` |
| `v3` | SceneWalk-like event structure and recall | `0.5836` | `0.5933` | `0.5750` | `0.2256` | `0.0878` |
| `v4` | `v3` structure with stricter audio restraint | `0.5830` | `0.6006` | `0.5671` | `0.2284` | `0.0886` |

Interpretation:

- `v3` produced the best unaggregated BERTScore F1
- `v4` improved SODA and ROUGE-L slightly, but lost enough recall that its BERTScore F1 was slightly lower than `v3`
- prompt wording alone improved the baseline, but did not reach the `0.6` to `0.7` target range

### Stage 4: Problem Identified - Another Recommendation

The next problem was temporal granularity mismatch. Kairos creates many short scene descriptions, while SceneWalk references describe broader event spans:

- video `mDvkux01G3A`: `301` Kairos scenes vs `80` SceneWalk segments
- video `X9MAf245Yag`: `166` Kairos scenes vs `51` SceneWalk segments

The recommendation was to test fixed, reference-independent temporal aggregation. This was justified as an evaluation-alignment step because it uses only Kairos timestamps, Kairos scene order, a fixed window size, and a fixed max-gap threshold.

### Stage 5: Recommendation Tried - New Table

Fixed-window scene merging was tested without removing any prompt-version results. The best development configuration became `v4 + fixed_window 13s`.

| Configuration | Aggregation | Window | BERTScore F1 | Precision | Recall | ROUGE-L F1 | SODA F1 | Matched pairs |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `v3` output | none | none | `0.5836` | `0.5933` | `0.5750` | `0.2256` | `0.0878` | `113` |
| `v4` output | none | none | `0.5830` | `0.6006` | `0.5671` | `0.2284` | `0.0886` | `113` |
| `v3` output | `fixed_window` | `10s` | `0.5861` | not recorded | not recorded | `0.2311` | `0.1244` | `126` |
| `v3` output | `fixed_window` | `13s` | `0.5851` | not recorded | not recorded | `0.2318` | `0.1429` | `129` |
| `v3` output | `fixed_window` | `15s` | `0.5841` | not recorded | not recorded | `0.2294` | `0.1528` | `129` |
| `v3` output | `fixed_window` | `20s` | `0.5835` | not recorded | not recorded | `0.2207` | `0.1722` | `129` |
| `v3` output | `fixed_window` | `25s` | `0.5825` | not recorded | not recorded | `0.2127` | `0.1792` | `128` |
| `v3` output | `fixed_window` | `30s` | `0.5800` | not recorded | not recorded | `0.2041` | `0.1840` | `124` |
| `v3` output | `fixed_window` | `45s` | `0.5784` | not recorded | not recorded | `0.1884` | `0.1521` | `91` |
| `v4` output | `fixed_window` | `8s` | `0.5841` | not recorded | not recorded | `0.2305` | `0.1065` | `121` |
| `v4` output | `fixed_window` | `10s` | `0.5862` | not recorded | not recorded | `0.2346` | `0.1258` | `126` |
| `v4` output | `fixed_window` | `12s` | `0.5878` | not recorded | not recorded | `0.2357` | `0.1397` | `129` |
| `v4` output | `fixed_window` | `13s` | `0.5879` | `0.5938` | `0.5834` | `0.2358` | `0.1450` | `129` |
| `v4` output | `fixed_window` | `14s` | `0.5873` | `0.5908` | `0.5850` | `0.2366` | `0.1509` | `130` |

Interpretation:

- fixed-window aggregation improved BERTScore F1 from `0.5836` to `0.5879`
- fixed-window aggregation improved SODA F1 from `0.0886` to `0.1450`
- the `13s` window gave the best BERTScore tradeoff; wider windows improved SODA but made text too broad for semantic matching

### Stage 6: Problem Identified - Another Recommendation

The fixed-window policy improved temporal alignment, but it still did not reach `0.6` BERTScore. The problem identified was that simple concatenation can create long, repetitive prediction text. The next recommendation was to try a benchmark-only abstractive aggregation rewrite.

This step was methodologically limited:

- use the best fixed-window policy: `v4 + fixed_window 13s`
- use only Kairos predicted descriptions and timestamps
- do not use SceneWalk reference captions
- save the rewrite prompt separately as `prompts/benchmark_versions/aggregate_scene_segments_v1.txt`
- test only on the same 2-video development subset

### Stage 7: Recommendation Tried - New Results Added

The abstractive aggregation rewrite was run on both development videos using:

```bash
python test/benchmarks/run_scenewalk_benchmark.py --max-videos 2 --skip-pipeline --output-cache-name scenewalk_outputs --aggregate-predictions fixed_window --aggregation-window-sec 13 --aggregation-max-gap-sec 5 --rewrite-aggregates --aggregation-rewrite-prompt prompts/benchmark_versions/aggregate_scene_segments_v1.txt --rewrite-max-workers 6
```

Result file:

- `test/benchmarks/results/scenewalk_results_20260606_110228.json`

| Configuration | Aggregation | Rewrite Prompt | BERTScore F1 | Precision | Recall | ROUGE-L F1 | SODA F1 | Matched pairs |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Current best before rewrite | `fixed_window 13s` | none | `0.5879` | `0.5938` | `0.5834` | `0.2358` | `0.1450` | `129` |
| Rewrite trial | `fixed_window 13s` | `aggregate_scene_segments_v1.txt` | `0.5777` | `0.6136` | `0.5468` | `0.2063` | `0.1273` | `129` |

Per-video rewrite completion:

| Video | Raw Kairos Scenes | Rewritten Aggregated Segments | SceneWalk GT Segments | Matched Pairs | SODA F1 |
|---|---:|---:|---:|---:|---:|
| `mDvkux01G3A` | `301` | `193` | `80` | `78` | `0.1143` |
| `X9MAf245Yag` | `166` | `106` | `51` | `51` | `0.1403` |

Interpretation:

- the rewrite trial completed on both development videos
- abstractive rewriting did not improve the current best result
- BERTScore F1 dropped from `0.5879` to `0.5777`
- precision increased, but recall dropped sharply, which suggests the rewrite became too conservative or removed useful details
- ROUGE-L and SODA also dropped, so `aggregate_scene_segments_v1.txt` should not be frozen
- the current best development configuration remains `v4 + fixed_window 13s`

Current recommendation:

Freeze neither the rewrite prompt nor any new prompt based on this result. If the project must push closer to `0.6`, the next non-overfitting option is error analysis on `scenewalk_comparison.md` followed by one small, predeclared aggregation-policy change, not another broad rewrite of the scene prompt.
