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

## Next Actions

1. Finalize the tuned prompts.
2. Mark current 2-video SceneWalk results as development runs.
3. Run held-out SceneWalk evaluation on additional unseen videos.
4. Expand TIB to a larger held-out sample.
5. Add one short error-analysis paragraph to the paper using examples from `scenewalk_comparison.md`.
