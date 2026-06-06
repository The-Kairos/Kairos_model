# Kairos Benchmarking Final Results Summary

This note summarizes the current development benchmarking outcome for Kairos on the 2-video SceneWalk development subset. It is shorter than `BENCHMARKING_PUBLICATION_NOTE.md` and is intended to guide the final paper structure.

## Current Best Development Configuration

The current best development configuration is:

- base prompt/output source: `v4`
- aggregation policy: `fixed_window`
- aggregation window: `13s`
- max gap between merged scenes: `5s`
- result file: `test/benchmarks/results/scenewalk_results_20260606_104931.json`

| Configuration | BERTScore F1 | Precision | Recall | ROUGE-L F1 | SODA F1 |
|---|---:|---:|---:|---:|---:|
| Original pre-versioning baseline | `0.5723` | `0.5212` | `0.6352` | `0.1771` | `0.0686` |
| Best unaggregated prompt result: `v3` | `0.5836` | `0.5933` | `0.5750` | `0.2256` | `0.0878` |
| Current best: `v4 + fixed_window 13s` | `0.5879` | `0.5938` | `0.5834` | `0.2358` | `0.1450` |

Compared with the original pre-versioning result:

- BERTScore F1 improved from `0.5723` to `0.5879` (`+0.0156`)
- BERTScore precision improved from `0.5212` to `0.5938` (`+0.0726`)
- ROUGE-L F1 improved from `0.1771` to `0.2358` (`+0.0587`)
- SODA F1 improved from `0.0686` to `0.1450` (`+0.0764`)
- BERTScore recall decreased from `0.6352` to `0.5834` (`-0.0518`)

The recall drop is important to report internally: the tuned prompts became more precise and grounded, but less broad than the original baseline. The overall result is still stronger because semantic precision, lexical overlap, and temporal alignment all improved.

## Frozen Held-Out Evaluation Result

The selected configuration was frozen and evaluated on two held-out SceneWalk videos that were excluded from the development prompt/aggregation tuning:

- held-out videos: `c0VPJWt_f0w`, `NkMWgw6hNrE`
- excluded development videos: `mDvkux01G3A`, `X9MAf245Yag`
- result file: `test/benchmarks/results/scenewalk_results_20260606_154048.json`
- generated held-out report: `test/benchmarks/results/scenewalk_benchmark_report.md`
- generated held-out comparison: `test/benchmarks/results/scenewalk_comparison.md`
- dedicated qualitative comparison: `log_reports/scenewalk_heldout_description_comparison.md`

| Configuration | BERTScore F1 | Precision | Recall | ROUGE-L F1 | SODA F1 | Matched Pairs |
|---|---:|---:|---:|---:|---:|---:|
| Frozen held-out: `v4 + fixed_window 13s` | `0.5886` | `0.5885` | `0.5903` | `0.2305` | `0.1382` | `129` |

This held-out result is slightly higher than the development best on BERTScore F1 (`0.5879 -> 0.5886`) and remains close on ROUGE-L (`0.2358 -> 0.2305`) and SODA (`0.1450 -> 0.1382`). This supports the claim that the selected prompt plus aggregation policy did not obviously overfit the two development videos.

## Why Prompt Tuning Is Justified

Prompt tuning is valid here because Kairos uses an LLM fusion stage to convert multimodal evidence into scene descriptions. The prompt is therefore part of the system configuration, not an external scoring trick.

The tuning was methodologically controlled:

- each prompt version was saved under `prompts/benchmark_versions/`
- tested prompt versions were not edited in place
- results were logged for every prompt version
- the 2-video SceneWalk run was treated as a development subset only
- prompt changes were justified as improving grounded visual description and multimodal fusion
- the prompts were not written using SceneWalk reference captions

In paper terms, this can be described as development-set prompt selection or prompt ablation, not as final benchmark evidence.

## Why Temporal Aggregation Is Justified

Temporal aggregation is justified because SceneWalk and Kairos use different evaluation units.

On the development subset:

- video `mDvkux01G3A`: Kairos produced `301` raw scenes, while SceneWalk had `80` ground-truth segments
- video `X9MAf245Yag`: Kairos produced `166` raw scenes, while SceneWalk had `51` ground-truth segments

This means Kairos often describes a shorter visual moment while SceneWalk references describe a broader event span. A correct Kairos scene can score poorly if it is compared to a longer reference that includes neighboring actions.

The selected aggregation policy is defensible because it:

- uses only Kairos timestamps and scene order
- uses fixed parameters selected on the development subset
- does not use SceneWalk reference text during merging
- is applied before scoring
- will be frozen before held-out evaluation

This should be framed as evaluation alignment, not as changing the underlying Kairos model.

## What Was Tried But Not Selected

An abstractive aggregation rewrite was also tested:

- prompt: `prompts/benchmark_versions/aggregate_scene_segments_v1.txt`
- result file: `test/benchmarks/results/scenewalk_results_20260606_110228.json`

| Configuration | BERTScore F1 | Precision | Recall | ROUGE-L F1 | SODA F1 |
|---|---:|---:|---:|---:|---:|
| `v4 + fixed_window 13s` | `0.5879` | `0.5938` | `0.5834` | `0.2358` | `0.1450` |
| `v4 + fixed_window 13s + rewrite_v1` | `0.5777` | `0.6136` | `0.5468` | `0.2063` | `0.1273` |

The rewrite increased precision but reduced recall, ROUGE-L, and SODA. It should not be frozen. The likely issue is that the rewrite made captions cleaner but removed useful details needed for SceneWalk matching.

## Is This Valid To Publish?

Yes, the methodology is publishable if the paper clearly separates development selection from held-out evaluation.

The development results should not be presented as the final benchmark score. They are valid as ablations that justify selecting a prompt and aggregation policy.

Industry and research-standard requirements:

- define a development subset for prompt and policy selection
- freeze the selected prompt and aggregation policy
- run final evaluation on unseen videos
- report held-out results as the main benchmark result
- include development ablations only as supporting evidence
- avoid changing prompts or aggregation after looking at held-out results

The current work satisfies the development-stage requirements. The next publishable step is held-out SceneWalk evaluation with the frozen `v4 + fixed_window 13s` configuration.

## What To Put In The Paper

For the main results section, include:

- held-out SceneWalk results using the frozen configuration
- BERTScore F1, precision, and recall
- SODA F1, precision, and recall
- ROUGE-L F1
- number of held-out videos and matched pairs
- a short note that temporal aggregation is fixed and reference-independent

For the methods section, include:

- Kairos generates scene descriptions from multimodal evidence
- prompt selection was performed on a small development subset
- final prompt and aggregation policy were frozen before held-out evaluation
- fixed-window aggregation aligns Kairos scene units with SceneWalk segment units
- aggregation uses timestamps and scene order only, not reference captions

For the ablation section or appendix, include:

- original baseline
- `v1` to `v4` prompt ablation table
- fixed-window aggregation table
- failed abstractive rewrite result, if space allows

If the paper has limited space, do not include every failed trial in the main results table. Put the successful frozen held-out result in the main table, and move development ablations to an appendix or short ablation subsection.

## Qualitative Narrative For The Paper

The recommended interpretation is not that low overlap scores mean Kairos descriptions are worse. A more accurate and publishable framing is:

```text
Automated overlap metrics can understate Kairos quality because Kairos often produces more temporally local, visually specific descriptions than SceneWalk's broader segment captions. The mismatch is not only description quality, but annotation granularity and reference style.
```

This narrative is useful because SceneWalk captions often summarize a broad event span, while Kairos describes shorter scene units with concrete visual evidence. That can lower BERTScore, ROUGE-L, and SODA even when the Kairos output is useful, grounded, and more detailed for downstream retrieval or question answering.

The paper should show this with examples rather than only stating it. Good example candidates from the current development comparison are:

| Video | Kairos Scene / Time | Why It Is Useful |
|---|---|---|
| `mDvkux01G3A` | scene `116`, around `1317.6-1328.5s` | Kairos gives concrete beach, couch, window, sunset, and bench details while the SceneWalk reference is broader and organized differently. |
| `X9MAf245Yag` | scene `33`, around `381.9-394.1s` | Kairos describes visible actions and objects such as window blinds, a Wii controller, painting, and cutting, while the reference summarizes a wider event sequence. |
| `X9MAf245Yag` | scene `10`, around `94.1-105.8s` | Kairos includes clothing, outlet, dim-room, and wall details that help show visual specificity beyond simple overlap scoring. |

When writing these examples, avoid claiming that Kairos is always better than SceneWalk. The stronger claim is that the two systems describe different units of analysis: SceneWalk is segment-level and reference-style, while Kairos is scene-level and evidence-rich.

## Where The Description Files Are

SceneWalk ground-truth descriptions:

- `test/benchmarks/cache/scenewalk_manifest.json`
- `test/benchmarks/cache/scenewalk_heldout_manifest.json`
- `test/benchmarks/results/scenewalk_comparison.md`
- `test/benchmarks/results/scenewalk_comparison.json`

Kairos development outputs:

- current `v4`: `test/benchmarks/cache/scenewalk_outputs/video_000/checkpoint.json`
- current `v4`: `test/benchmarks/cache/scenewalk_outputs/video_001/checkpoint.json`
- `v1` backups: `test/benchmarks/cache/scenewalk_outputs_v1_backup_20260606/`
- `v2` backups: `test/benchmarks/cache/scenewalk_outputs_v2_backup_20260606/`
- `v3` backups: `test/benchmarks/cache/scenewalk_outputs_v3_backup_20260606/`

Prompt files:

- active full scene prompt: `prompts/describe_scene.txt`
- active fallback prompt: `prompts/fallback_describe_scene.txt`
- active short scene prompt: `prompts/describe_scene_short.txt`
- versioned prompt archive: `prompts/benchmark_versions/`

## Recommended Paper Structure

1. Method:
   - describe Kairos pipeline and scene-description generation
   - describe SceneWalk evaluation setup
   - describe metrics: BERTScore, SODA, ROUGE-L

2. Development selection:
   - briefly state that prompt versions and aggregation policy were selected on a 2-video development subset
   - include the compact ablation table if space allows

3. Final held-out evaluation:
   - report only frozen-pipeline held-out results as the main benchmark score
   - clearly label the number of videos and whether aggregation was used

4. Error analysis:
   - include one paragraph from `scenewalk_comparison.md`
   - focus on temporal granularity mismatch, missing visual details, and over/under-specific descriptions

5. Limitations:
   - development subset was small
   - prompt tuning can improve formatting but cannot fully solve missing visual evidence
   - held-out results are the real evidence of generalization

## Current Decision

Do not freeze the abstractive rewrite prompt. The strongest development choice remains:

```text
describe_scene_v4 + fallback_describe_scene_v4 + describe_scene_short_v4
with fixed_window aggregation, 13s window, 5s max gap
```

This configuration should now be frozen and evaluated on held-out SceneWalk videos.
