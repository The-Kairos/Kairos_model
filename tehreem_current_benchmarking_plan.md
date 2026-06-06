# Tehreem Current Benchmarking Plan

## Mission

Produce benchmarking results that are strong enough to publish and defensible enough to survive reviewer scrutiny.

This plan exists to keep the benchmarking work:
- reversible
- auditable
- methodologically clean
- aligned with the paper goal

The central rule is:
- do not tune on the same benchmark videos that we later present as final held-out test results

## Plain-Language Goal

We are trying to do the following:

1. Improve the prompt that turns Kairos multimodal scene evidence into a scene description.
2. Test that prompt on a small development set.
3. Decide on one final prompt version.
4. Freeze that prompt.
5. Run final benchmark results on unseen videos only.
6. Report those unseen-video numbers in the paper.

That is the publishable workflow.

## Why This Plan Exists

Without a written plan, prompt tuning can become messy very quickly:
- we may forget which prompt version changed what
- we may accidentally use tuned-on data as final test evidence
- we may lose track of why a prompt was changed
- we may not be able to explain the methodology cleanly in the paper

This file is therefore the working source of truth for the benchmarking push.

## Prompt Versioning Rule

From this point onward:
- never edit an existing benchmark prompt in place once it has been used for a recorded benchmark run
- every new prompt iteration must be saved as a new versioned prompt file
- every tested prompt version must have its benchmark results recorded in this plan
- the final frozen prompt must be chosen from the recorded results history

Why this rule exists:
- it preserves exact version history
- it makes prompt experiments reversible
- it prevents loss of strong prompt candidates
- it makes the paper methodology easier to explain
- it reduces the chance of accidental undocumented overfitting

## Current Benchmark Reality

### SceneWalk

Current checked-in development result:
- `Matched BERTScore F1`: `0.5723`
- `SODA F1`: `0.0686`
- report file: `test/benchmarks/results/scenewalk_benchmark_report.md`

Important interpretation:
- these current 2-video SceneWalk results must now be treated as development results
- they should not be presented as final held-out benchmark numbers if we keep tuning prompts against them

### TIB

Current checked-in result:
- `BERTScore F1`: `0.5931`
- report file: `test/benchmarks/results/tib_benchmark_report.md`

Important interpretation:
- TIB is a separate benchmark task
- it evaluates full-video synopsis quality, not scene-level description alignment
- TIB should be expanded later on a larger held-out sample

## High-Level Benchmarking Strategy

### Stage A: Development / Prompt Tuning

Use the current small SceneWalk run as a development set for prompt iteration.

Purpose:
- identify recurring failure modes
- modify prompt wording
- rerun benchmark
- compare scores and outputs

Output of this stage:
- one prompt version that looks best overall

### Stage B: Freeze Prompt

Once the development evidence is good enough:
- stop editing prompts
- record that prompt version as frozen
- use that exact version for held-out evaluation

Output of this stage:
- fixed benchmark configuration

### Stage C: Held-Out Final Evaluation

Run the frozen prompt on unseen SceneWalk videos and larger held-out TIB samples.

Purpose:
- get final publishable metrics
- avoid benchmark leakage
- ensure the paper reports results from unseen examples

Output of this stage:
- final paper-ready benchmark numbers

## Files Already Changed

These files have already been changed in the repo as part of the first tuning direction.

### Prompt Files

- `prompts/describe_scene.txt`
- `prompts/fallback_describe_scene.txt`
- `prompts/describe_scene_short.txt`

### Documentation Files

- `log_reports/BENCHMARKING_PUBLICATION_NOTE.md`

Important note:
- the currently edited prompt files are now the active working copy
- from the next prompt iteration onward, all benchmark prompt changes should move into versioned prompt files instead of repeatedly overwriting the same prompt

## Why These Prompt Files Matter

### `prompts/describe_scene.txt`

Purpose:
- main prompt for generating the final scene description from Kairos multimodal evidence

Why it matters:
- this is the primary scene-fusion prompt
- it directly affects SceneWalk evaluation
- it also indirectly affects TIB because scene descriptions feed the narrative and synopsis stages

Current tuning direction already applied:
- reduce decorative language
- reduce unsupported specifics
- emphasize visible actions and settings
- keep output more compact and grounded

What to watch for:
- whether BERTScore improves
- whether SODA improves or stays stable
- whether outputs become too short and lose important details

### `prompts/fallback_describe_scene.txt`

Purpose:
- fallback prompt when the scene evidence is sparse or noisy

Why it matters:
- weak scenes can poison the benchmark if they become speculative
- the fallback prompt must stay aligned with the main prompt

Current tuning direction already applied:
- prioritize conservative grounded descriptions
- reduce guesses based on weak evidence
- keep sparse scenes compact and reliable

What to watch for:
- whether fallback scenes become too generic
- whether they still preserve enough content to match benchmark references semantically

### `prompts/describe_scene_short.txt`

Purpose:
- generates short scene summaries used as context before the final scene description stage

Why it matters:
- this prompt influences the context passed into final descriptions
- overly speculative short summaries can contaminate later scene outputs

Current tuning direction already applied:
- keep summaries short
- focus on main visible events
- reduce interpretive and cinematic language

What to watch for:
- whether context becomes too thin
- whether shorter context improves groundedness in later scene descriptions

## Core Pipeline Files To Keep In Mind

These files are not necessarily changed immediately, but they are central to understanding the benchmark behavior.

### `src/scene_description.py`

Why important:
- this is where prompt files are loaded and used
- this is the scene-description assembly path
- this is where short summaries and final scene descriptions interact

What to inspect:
- prompt paths
- fallback behavior
- raw scene formatting
- how audio is appended to formatted scene text

### `main.py`

Why important:
- runs `describe_scenes_log(...)`
- connects scene description generation to the full pipeline
- controls how descriptions flow into narrative and synopsis generation

What to inspect:
- the `describe_scenes_log(...)` call
- benchmark execution flow
- whether there are other parameters that may influence the benchmark

### `test/benchmarks/run_scenewalk_benchmark.py`

Why important:
- main SceneWalk evaluation runner
- computes benchmark metrics
- generates report files

What to inspect:
- how current dev videos are selected
- how matched pairs are created
- how outputs are saved

### `test/benchmarks/metrics/soda_metric.py`

Why important:
- contains SODA logic
- necessary if we need to explain SODA clearly in the paper

### `test/benchmarks/run_tib_benchmark.py`

Why important:
- main TIB evaluation runner
- useful later when we expand the held-out TIB sample

## Existing Benchmark Artifacts We Must Preserve

These are important because they form the baseline for comparison.

### Baseline Development Results

- `test/benchmarks/results/scenewalk_benchmark_report.md`
- `test/benchmarks/results/scenewalk_comparison.md`
- `test/benchmarks/results/scenewalk_comparison.json`
- `test/benchmarks/results/tib_benchmark_report.md`
- `test/benchmarks/results/tib_comparison.md`
- `test/benchmarks/results/tib_comparison.json`

Why preserve them:
- they are the pre-tuning or early-tuning reference point
- we need them to compare before/after prompt changes
- they help build the paper benchmark narrative

## Folder Tracking

This section lists folders we rely on or may add to during the benchmarking push.

### Existing Important Folders

- `prompts/`
- `src/`
- `test/benchmarks/`
- `test/benchmarks/results/`
- `test/benchmarks/cache/`
- `log_reports/`

### Folders To Add Only If Needed

- `prompts/benchmark_versions/`
Purpose:
- store versioned benchmark prompt files such as `describe_scene_v1.txt`, `fallback_describe_scene_v1.txt`, and `describe_scene_short_v1.txt`

- `test/benchmarks/results/dev_runs/`
Purpose:
- archive development-only SceneWalk runs separately from final held-out runs

- `test/benchmarks/results/final_runs/`
Purpose:
- hold frozen-prompt held-out results for clean paper reporting

Note:
- these folders are not required yet, but if we start generating multiple official comparison runs, separating them is strongly recommended

## Methodology Rules We Must Not Break

### Rule 1

Do not present prompt-tuned development runs as final benchmark evidence.

### Rule 2

Every prompt change must be justified in terms of:
- groundedness
- better multimodal fusion
- reduced hallucination
- more faithful scene rendering

Not:
- make it look better at any cost

### Rule 2.5

Do not overfit to the exact wording style of the 2-video SceneWalk development subset.

What this means:
- we can improve groundedness, compactness, and multimodal faithfulness
- we can aim to improve `BERTScore`, `SODA`, and the other benchmark metrics used in development
- we should not chase shallow phrasing tricks that only help these two videos
- the chosen prompt should still look like a general Kairos scene-description method, not a SceneWalk-only wording hack

### Rule 3

Keep a frozen final prompt before running unseen held-out videos.

### Rule 4

In the paper, disclose prompt tuning as part of the scene-fusion design.

### Rule 5

Treat `BERTScore` as the primary semantic metric, `SODA` as temporal alignment, and lexical metrics as secondary.

## Prompt-Tuning Checklist

This checklist is the active working area.

### Phase 1: Document Current State

- [x] Confirm SceneWalk benchmark exists
- [x] Confirm `SODA` is already used
- [x] Confirm current SceneWalk scores
- [x] Confirm current TIB scores
- [x] Identify prompt files that affect scene descriptions
- [x] Save publication/methodology note
- [x] Create this working benchmarking plan

### Phase 2: First Prompt-Tuning Pass

- [x] Tighten `prompts/describe_scene.txt`
- [x] Tighten `prompts/fallback_describe_scene.txt`
- [x] Tighten `prompts/describe_scene_short.txt`
- [x] Shift prompt style toward compact grounded descriptions
- [x] Reduce unsupported specifics and cinematic filler

Reason for this phase:
- current SceneWalk comparison shows Kairos is often too long, too decorative, and too specific
- this hurts alignment with benchmark references

### Phase 2.5: Start Prompt Version History

- [x] Create `prompts/benchmark_versions/`
- [x] Save the current main prompt as a versioned file
- [x] Save the current fallback prompt as a versioned file
- [x] Save the current short prompt as a versioned file
- [x] Choose a stable naming convention for all future prompt versions
- [x] Record the first versioned prompt set in the prompt ledger below

Suggested naming convention:
- `prompts/benchmark_versions/describe_scene_v1.txt`
- `prompts/benchmark_versions/fallback_describe_scene_v1.txt`
- `prompts/benchmark_versions/describe_scene_short_v1.txt`

Reason for this phase:
- we do not want to lose version history
- we do not want undocumented prompt drift
- we want every tested prompt and result to remain recoverable

### Phase 3: Evaluate Tuned Prompt On Development Set

- [x] Clear SceneWalk benchmark cached outputs so prompt changes take effect
- [x] Rerun SceneWalk on the current development set
- [x] Save the resulting report files
- [x] Compare old and new `Matched BERTScore F1`
- [x] Compare old and new `SODA F1`
- [x] Compare old and new `Matched ROUGE-L F1`
- [ ] Compare old and new `BERTScore Precision`
- [ ] Compare old and new `BERTScore Recall`
- [ ] Read `scenewalk_comparison.md` again to inspect qualitative change
- [x] Decide if the new prompt is better, worse, or mixed
- [x] Record the tested prompt version and all resulting metrics in this plan

Files involved:
- `test/benchmarks/cache/scenewalk_outputs/`
- `test/benchmarks/results/scenewalk_benchmark_report.md`
- `test/benchmarks/results/scenewalk_comparison.md`
- `test/benchmarks/results/scenewalk_comparison.json`

Reason for this phase:
- we need evidence that prompt edits actually help
- we should not freeze a prompt until we verify improvement

Development decision standard:
- prefer the prompt version that improves `Matched BERTScore F1` while keeping `SODA` healthy
- use `ROUGE-L`, precision, and recall as supporting signals
- do not freeze a prompt that looks numerically better only because it mimics surface wording on the 2-video dev set

### Phase 4: Second Prompt-Tuning Pass If Needed

- [ ] Identify remaining failure modes from dev rerun
- [ ] Make only targeted prompt edits in a new versioned prompt file
- [ ] Record exactly what changed
- [ ] Rerun development benchmark again
- [ ] Compare with prior dev run
- [ ] Record the new prompt version and full metric results in this plan

Possible failure modes to watch:
- still too verbose
- too generic
- audio dominating visual content
- unsupported detail still present
- not enough action emphasis
- scene boundaries causing mismatch

Reason for this phase:
- prompt tuning should be iterative but controlled
- each iteration must have a reason

### Phase 5: Freeze Prompt

- [ ] Decide best versioned main prompt
- [ ] Decide best versioned fallback prompt
- [ ] Decide best versioned short prompt
- [ ] Choose the prompt set with the best overall development evidence
- [ ] Mark the development prompt as frozen
- [ ] Stop changing prompt wording before held-out evaluation
- [ ] Copy the chosen frozen version into the active prompt paths only after the freeze decision

Reason for this phase:
- a frozen prompt is required for clean held-out testing

### Phase 6: Held-Out SceneWalk Evaluation

- [ ] Select additional unseen SceneWalk videos
- [ ] Confirm they were not used in prompt tuning
- [ ] Run frozen prompt on held-out SceneWalk set
- [ ] Save held-out reports separately
- [ ] Treat only these held-out numbers as final paper results

Files involved:
- `test/benchmarks/run_scenewalk_benchmark.py`
- `test/benchmarks/results/`

Reason for this phase:
- this is the actual publishable SceneWalk result

### Phase 7: Held-Out TIB Expansion

- [ ] Increase TIB sample size
- [ ] Keep the prompt frozen if we are using the same evaluation configuration
- [ ] Save expanded TIB report
- [ ] Report TIB as the full-video synopsis benchmark

Files involved:
- `test/benchmarks/run_tib_benchmark.py`
- `test/benchmarks/results/tib_benchmark_report.md`

Reason for this phase:
- TIB provides a separate benchmark claim
- more videos make the result more defensible

### Phase 8: Paper Integration

- [ ] Add benchmarking protocol paragraph
- [ ] Add prompt-tuning disclosure paragraph
- [ ] Add one short error-analysis paragraph
- [ ] Explain why `BERTScore` is the primary metric
- [ ] Explain what `SODA` contributes
- [ ] Clearly label development vs held-out evaluation
- [ ] State that prompt versions were tracked and selected using development-only results

Reason for this phase:
- even good benchmark numbers can look weak if the methodology section is vague

## Revertibility Notes

To keep this reversible:
- keep every prompt edit small and intentional
- do not mix prompt edits with unrelated code changes
- never destroy a tested prompt version
- compare benchmark results after each prompt pass
- preserve old report files before overwriting them if possible
- save every prompt iteration as a new versioned prompt file

If needed, the easiest items to revert are:
- `prompts/benchmark_versions/describe_scene_v*.txt`
- `prompts/benchmark_versions/fallback_describe_scene_v*.txt`
- `prompts/benchmark_versions/describe_scene_short_v*.txt`

## Prompt Version Ledger

This ledger must be updated every time a prompt version is tested.

### How To Use The Ledger

For each tested prompt version, record:
- version label
- exact prompt files used
- why the version was created
- benchmark date
- all development metrics
- qualitative takeaway
- whether it is still a freeze candidate

### Current Baseline / Pre-Versioning Reference

- version label: `pre_versioning_dev_baseline`
- prompt files used:
- `prompts/describe_scene.txt`
- `prompts/fallback_describe_scene.txt`
- `prompts/describe_scene_short.txt`
- benchmark scope:
- current 2-video SceneWalk development set
- metrics:
- `Matched BERTScore F1`: `0.5723`
- `BERTScore Precision`: `0.5212`
- `BERTScore Recall`: `0.6352`
- `Matched ROUGE-L F1`: `0.1771`
- `SODA F1`: `0.0686`
- `SODA Precision`: `0.0443`
- `SODA Recall`: `0.1523`
- qualitative takeaway:
- baseline development reference before formal prompt version tracking
- freeze candidate:
- no

### Future Prompt Test Entry Template

- version label:
- prompt files used:
- benchmark date:
- reason for change:
- `Matched BERTScore F1`:
- `BERTScore Precision`:
- `BERTScore Recall`:
- `Matched ROUGE-L F1`:
- `SODA F1`:
- `SODA Precision`:
- `SODA Recall`:
- qualitative takeaway:
- freeze candidate:

### Prompt Candidate `v1`

- version label: `v1`
- prompt files used:
- `prompts/benchmark_versions/describe_scene_v1.txt`
- `prompts/benchmark_versions/fallback_describe_scene_v1.txt`
- `prompts/benchmark_versions/describe_scene_short_v1.txt`
- benchmark date:
- `2026-06-06`
- reason for change:
- first formal versioned prompt set derived from the initial grounded-and-compact tuning pass
- metrics:
- `Matched BERTScore F1`: `0.5745`
- `Matched ROUGE-L F1`: `0.2208`
- `SODA F1`: `0.0859`
- `SODA Precision`: `0.0553`
- `SODA Recall`: `0.1925`
- `Total matched pairs`: `113`
- baseline comparison:
- `Matched BERTScore F1`: `0.5723 -> 0.5745` (`+0.0022`)
- `Matched ROUGE-L F1`: `0.1771 -> 0.2208` (`+0.0437`)
- `SODA F1`: `0.0686 -> 0.0859` (`+0.0173`)
- qualitative takeaway:
- `v1` is a real improvement over the pre-versioning baseline
- the biggest gains are in `SODA` and `ROUGE-L`
- `BERTScore` improved slightly, but not enough to justify freezing without at least one more controlled prompt iteration
- freeze candidate:
- yes, but weak

## Decision Standard For Freezing The Prompt

We should freeze the prompt if most of the following are true:
- `Matched BERTScore F1` improves meaningfully on the dev set
- `SODA F1` does not collapse
- precision and recall remain reasonable rather than becoming skewed
- outputs look more grounded and less over-described
- qualitative mismatches in `scenewalk_comparison.md` decrease
- the prompt does not become so generic that descriptions lose useful content
- the prompt looks generally valid, not just narrowly optimized for the 2 dev videos

We should not freeze the prompt if:
- BERTScore improves but descriptions become obviously weak or generic
- SODA drops substantially because descriptions no longer map well to scenes
- outputs become shorter but less informative
- the prompt still produces many unsupported specifics
- the wording looks like it is optimized only to mirror SceneWalk phrasing rather than improve general multimodal scene description

## Most Important Immediate Next Step

The next action is not to invent more benchmark theory.

The next action is:
- create versioned copies of the current tuned prompts
- record them as the first official prompt candidate
- rerun the development SceneWalk benchmark with that versioned prompt set
- compare the new results to the existing development baseline
- decide whether another prompt pass is needed before freezing

## Working Summary

In simple words, our mission is:
- tune the prompt on a dev set
- keep every version and every result
- pick the best prompt based on recorded metrics
- freeze it
- run unseen evaluation
- report only unseen results in the paper

That is how we get publishable benchmarking results without undermining the methodology.
