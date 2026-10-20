"""Summarize a saved run without making API calls."""
import argparse
import json
from pathlib import Path
import re
import statistics

from run_benchmark import aggregate


def timestamp_check(synopsis, narrative):
    def seconds(text):
        h, m, s = text.split(":")
        return round(int(h) * 3600 + int(m) * 60 + float(s), 3)
    source = {seconds(t) for t in re.findall(r"\b\d{2}:\d{2}:\d{2}(?:\.\d+)?\b", narrative)}
    emitted = []
    for key in ("video_highlights", "video_timeline", "suggested_clips"):
        for item in synopsis.get(key, []):
            for field in ("timestamp", "start", "end"):
                value = item.get(field, "")
                if re.fullmatch(r"\d{2}:\d{2}:\d{2}(?:\.\d+)?", str(value)):
                    emitted.append(seconds(value))
    return {"source_explicit_timestamp_count": len(source),
            "emitted_timestamp_fields": len(emitted),
            "timestamp_fields_absent_from_narrative": sum(t not in source for t in emitted)}


def analyze(directory):
    report = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    if report["status"] != "complete":
        raise ValueError("Benchmark is incomplete")
    rows = [json.loads(line) for line in (directory / "requests.jsonl").read_text(encoding="utf-8").splitlines()]
    narrative = json.loads(Path(report["input_path"]).read_bytes())["narratives"][-1]["narrative"]
    for run in report["runs"]:
        requests = [r for r in rows if r["run_id"] == run["run_id"] and r["kind"] == "request"]
        run["synopsis_usage"] = aggregate([r for r in requests if not r["stage"].startswith("scene:")])
        run["scene_usage"] = aggregate([r for r in requests if r["stage"].startswith("scene:")])
        # Seven primary calls are required. Additional calls are repairs, fills,
        # safe fallbacks, or retries, rather than proof of format success.
        run["additional_synopsis_requests"] = max(0, run["synopsis_usage"]["requests"] - 7)
        synopsis = json.loads((directory / run["run_id"] / "synopsis.json").read_text(encoding="utf-8"))
        run["quality"].update(timestamp_check(synopsis, narrative))
    report["aggregate_by_variant"] = {}
    for variant in ("original", "revised"):
        runs = [r for r in report["runs"] if r["variant"] == variant]
        matching = [r for r in rows if r.get("run_id", "").startswith(variant + "_") and r["kind"] == "request"]
        report["aggregate_by_variant"][variant] = {
            "synopsis_usage": aggregate([r for r in matching if not r["stage"].startswith("scene:")]),
            "scene_usage": aggregate([r for r in matching if r["stage"].startswith("scene:")]),
            "synopsis_wall_median_sec": statistics.median(r["synopsis_wall_sec"] for r in runs),
            "scene_wall_median_sec": statistics.median(r["sampled_scenes_wall_sec"] for r in runs),
            "warm_repeat_synopsis_median_sec": statistics.median(r["synopsis_wall_sec"] for r in runs if r["repetition"] > 1) if len(runs) > 1 else None,
        }
    (directory / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    repetitions = len({r['repetition'] for r in report['runs']})
    lines = ["# Saved-input prompt caching comparison", "",
             f"Model returned by Azure: `{report['response_model']}`. {repetitions} repetitions per layout, alternating order.",
             f"Saved final narrative and {report['scene_samples']} scene samples; no video models were rerun.", "",
             "| Run | Synopsis wall seconds | Scene sample wall seconds | Synopsis cached input | Scene cache hits | Extra synopsis calls |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for r in report["runs"]:
        usage = r["synopsis_usage"]
        fraction = usage["cached_token_fraction"]
        cache = f"{fraction:.1%}" if fraction is not None else "not reported"
        lines.append(f"| {r['run_id']} | {r['synopsis_wall_sec']:.2f} | {r['sampled_scenes_wall_sec']:.2f} | {cache} | {r['scene_usage']['cache_hit_requests']} | {r['additional_synopsis_requests']} |")
    lines += ["", "## Timing", ""]
    for variant, summary in report["aggregate_by_variant"].items():
        lines.append(f"- {variant}: median synopsis wall time {summary['synopsis_wall_median_sec']:.2f}s; median sampled-scene wall time {summary['scene_wall_median_sec']:.2f}s.")
    change = report["synopsis_wall_reduction_fraction"]
    lines += ["", f"Median synopsis wall-time reduction with revised layout: {change:.1%} (negative means slower).",
              "This is a small, noisy sample; token cache reuse alone does not establish a latency improvement.",
              "Repeated original prompts can also reuse their entire inputs. Compare first-use and later repeats separately.",
              "", "## Quality checks", "",
              "| Run | Required questions in order | Questions | Missing answers | Timestamp fields absent from source |",
              "| --- | --- | ---: | ---: | ---: |"]
    for r in report["runs"]:
        q = r["quality"]
        lines.append(f"| {r['run_id']} | {q['required_questions_in_order']} | {q['question_count']} | {q['missing_answer_count']} | {q['timestamp_fields_absent_from_narrative']}/{q['emitted_timestamp_fields']} |")
    lines += ["", "Timestamp checks compare emitted timecodes with source text; membership alone does not establish event alignment.",
              "Unknown answers must be reviewed against available source evidence. Counts alone do not distinguish justified abstentions from lost answers.",
              "Formatting checks do not establish factual quality. Raw section captures, when available, help distinguish generation from parsing failures.",
              "", "## Limits", ""] + ["- " + s for s in report["limitations"]]
    lines += ["", "Raw request records are in `requests.jsonl`; generated outputs are in each run directory.",
              "No prompt text, keys, endpoints, or response text are stored in request telemetry."]
    (directory / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report["aggregate_by_variant"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    analyze(parser.parse_args().directory)
