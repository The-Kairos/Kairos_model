"""Analyze recorded local full-LLM runs without additional API calls."""
import argparse
import json
from pathlib import Path
import random
import re
import statistics

from run_local_pipeline import ROOT, read_evidence
from run_benchmark import aggregate

STAGES = ("scene_descriptions", "narrative", "synopsis", "total_llm")


def group(stage):
    if stage.startswith("scene:"):
        return "scene_descriptions"
    if stage.startswith("narrative:"):
        return "narrative"
    return "synopsis"


def numeric_time(value):
    if not isinstance(value, str) or not re.fullmatch(r"\d{2}:\d{2}:\d{2}(?:\.\d{1,3})?", value):
        return None
    h, m, s = value.split(":")
    return round(int(h) * 3600 + int(m) * 60 + float(s), 3)


def check_outputs(result, evidence):
    synopsis = result["synopsis"]
    source_times = {numeric_time(scene.get(key)) for scene in evidence["scenes"]
                    for key in ("start_timecode", "end_timecode")}
    source_times.discard(None)
    counts = {}
    numeric = []
    unknown_times = 0
    for section, field in (("video_highlights", "highlight"), ("video_timeline", "event"), ("suggested_clips", "description")):
        entries = synopsis.get(section, [])
        counts[section] = sum(bool(item.get(field)) and str(item[field]).strip().lower().rstrip(".") != "not explicitly stated" for item in entries)
        for item in entries:
            for key in ("timestamp", "start", "end"):
                if key not in item:
                    continue
                value = numeric_time(item[key])
                if value is None:
                    unknown_times += 1
                else:
                    numeric.append(value)
    return {
        "scene_order_and_count_preserved": [s.get("scene_index") for s in result["scenes"]] == [s.get("scene_index") for s in evidence["scenes"]],
        "meaningful_descriptions": counts,
        "numeric_timestamp_fields": len(numeric),
        "unknown_timestamp_fields": unknown_times,
        "timestamp_fields_not_exactly_in_saved_evidence": sum(t not in source_times for t in numeric),
        "note": "Timestamp membership and description counts do not establish factual correctness or event alignment.",
    }


def paired_interval(ratios):
    """Exploratory interval; nine sequential pairs are not independent trials."""
    rng = random.Random(412)
    boot = sorted(statistics.median(rng.choices(ratios, k=len(ratios))) for _ in range(5000))
    return [boot[int(len(boot) * 0.025)], boot[int(len(boot) * 0.975)]]


def analyze(directory):
    report = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    if report["status"] != "complete":
        raise ValueError("The full local comparison has not completed")
    records = [json.loads(line) for line in (directory / "requests.jsonl").read_text(encoding="utf-8").splitlines()]
    for run in report["runs"]:
        requests = [r for r in records if r.get("run_id") == run["run_id"] and r["kind"] == "request"]
        run["usage_by_stage"] = {stage: aggregate([r for r in requests if group(r["stage"]) == stage]) for stage in STAGES[:-1]}
        result = json.loads((directory / run["run_id"] / "pipeline_outputs.json").read_text(encoding="utf-8"))
        evidence = read_evidence(ROOT / report["sources"][run["case"]]["path"].replace("\\", "/"))
        run["output_checks"] = check_outputs(result, evidence)
    report["comparisons"] = {}
    ratios = []
    for case in report["sources"]:
        runs = [r for r in report["runs"] if r["case"] == case]
        cohorts = {}
        for variant in ("original", "cache_friendly"):
            selected = [r for r in runs if r["variant"] == variant]
            requests = [r for r in records if r["kind"] == "request" and any(r.get("run_id") == s["run_id"] for s in selected)]
            cohorts[variant] = {
                "stage_wall_median_sec": {stage: statistics.median(r["wall_sec"][stage] for r in selected) for stage in STAGES},
                "first_trial_total_sec": selected[0]["wall_sec"]["total_llm"],
                "repeat_trial_total_median_sec": statistics.median(r["wall_sec"]["total_llm"] for r in selected[1:]) if len(selected) > 1 else None,
                "usage_by_stage": {stage: aggregate([r for r in requests if group(r["stage"]) == stage]) for stage in STAGES[:-1]},
            }
        paired = []
        for repetition in sorted({r["repetition"] for r in runs}):
            old = next(r for r in runs if r["repetition"] == repetition and r["variant"] == "original")
            new = next(r for r in runs if r["repetition"] == repetition and r["variant"] == "cache_friendly")
            ratio = new["wall_sec"]["total_llm"] / old["wall_sec"]["total_llm"]
            ratios.append(ratio)
            paired.append(dict(repetition=repetition, cache_friendly_over_original=ratio))
        report["comparisons"][case] = dict(cohorts=cohorts, paired_total_ratios=paired)
    report["paired_total_ratio_median"] = statistics.median(ratios)
    report["exploratory_paired_ratio_95pct_interval"] = paired_interval(ratios)
    report["interval_limit"] = "Small sequential sample with shared service/network conditions; this does not prove equivalence."
    (directory / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# Local complete LLM workflow comparison", "",
             f"Host recorded at run time: **{report['host']['node']}**, {report['host']['system']} {report['host']['release']}, Python {report['host']['python'].split()[0]}.",
             "Both layouts run on this same local machine and use the same configured Azure deployment. No VM is involved.",
             "All saved clues are reused; every scene report, intermediate narrative, and final report is generated anew.", "",
             "| Input | Scenes per run | Original total median | Cache-friendly total median | Difference |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for case, comparison in report["comparisons"].items():
        old, new = (comparison["cohorts"][v]["stage_wall_median_sec"]["total_llm"] for v in ("original", "cache_friendly"))
        lines.append(f"| {case} | {report['sources'][case]['scene_count']} | {old:.2f}s | {new:.2f}s | {(new/old-1):+.1%} |")
    lines += ["", "Positive differences mean slower. Medians of stages do not necessarily sum to the median total.", "",
              "## Stage wall times", "", "| Input | Stage | Original median | Cache-friendly median |", "| --- | --- | ---: | ---: |"]
    for case, comparison in report["comparisons"].items():
        for stage in STAGES[:-1]:
            old, new = (comparison["cohorts"][v]["stage_wall_median_sec"][stage] for v in ("original", "cache_friendly"))
            lines.append(f"| {case} | {stage} | {old:.2f}s | {new:.2f}s |")
    lines += ["", "## Actual cache reads", "", "| Input | Stage | Original cached input | Cache-friendly cached input |", "| --- | --- | ---: | ---: |"]
    for case, comparison in report["comparisons"].items():
        for stage in STAGES[:-1]:
            values = [comparison["cohorts"][v]["usage_by_stage"][stage]["cached_token_fraction"] for v in ("original", "cache_friendly")]
            text = [f"{v:.1%}" if v is not None else "not reported" for v in values]
            lines.append(f"| {case} | {stage} | {text[0]} | {text[1]} |")
    lines += ["", "The short-scene and map/reduce narrative prompts have no experimental rearrangement; their timings can change because upstream generated inputs change.",
              "The experimental layout rearranges full-scene instructions and shared-narrative synopsis prompts. A cache read is not proof of a faster complete stage.",
              "", "## First trials and repeat trials", "", "| Input | Layout | First trial | Later trial median |", "| --- | --- | ---: | ---: |"]
    for case, comparison in report["comparisons"].items():
        for variant, cohort in comparison["cohorts"].items():
            repeat = cohort["repeat_trial_total_median_sec"]
            lines.append(f"| {case} | {variant} | {cohort['first_trial_total_sec']:.2f}s | {repeat:.2f}s |" if repeat is not None else f"| {case} | {variant} | {cohort['first_trial_total_sec']:.2f}s | n/a |")
    lo, hi = report["exploratory_paired_ratio_95pct_interval"]
    lines += ["", f"Median paired total-time ratio: {report['paired_total_ratio_median']:.3f}. Exploratory bootstrap interval: [{lo:.3f}, {hi:.3f}]. A ratio below 1 means faster.",
              report["interval_limit"], "", "## Output checks", "",
              "| Run | Scenes kept | Questions | Fully missing required answers | Meaningful highlights/timeline/clips | Time fields not exactly in saved evidence |",
              "| --- | --- | ---: | ---: | --- | ---: |"]
    for run in report["runs"]:
        q, checks = run["quality"], run["output_checks"]
        counts = checks["meaningful_descriptions"]
        lines.append(f"| {run['run_id']} | {checks['scene_order_and_count_preserved']} | {q['question_count']} | {q['fully_missing_required_answers']} | {counts['video_highlights']}/{counts['video_timeline']}/{counts['suggested_clips']} | {checks['timestamp_fields_not_exactly_in_saved_evidence']} |")
    requests = [r for r in records if r["kind"] == "request"]
    lines += ["", f"API requests: {len(requests)}; failed attempts: {sum(bool(r.get('error_type')) for r in requests)}; application retry attempts: {sum(r.get('attempt',1)>1 for r in requests)}.",
              f"Tree-reduce calls: {sum(r['tree_reduce_requests'] for r in report['runs'])}. The configured production threshold is 50000 characters; reduction is not artificially forced.",
              "Timestamp membership checks do not prove event alignment. Missing answers may be justified when evidence is absent.",
              "Source-based factual spot review is recorded separately; structure alone does not prove equal quality.", "", "## Limits", ""] + ["- " + text for text in report["limitations"]]
    lines += ["- First-trial connection startup is not isolated; compare first and later trials separately.",
              "- No local transcription, detector, caption model, video decode, or embedding work is timed.",
              "- Raw outputs and request traces stay local. Host, package versions, file/input hashes, and full settings are recorded in metrics.json."]
    (directory / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"host": report["host"]["node"], "comparisons": {case: value["cohorts"] for case, value in report["comparisons"].items()}}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    analyze(parser.parse_args().directory)
