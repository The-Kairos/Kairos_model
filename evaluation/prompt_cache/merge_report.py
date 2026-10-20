"""Build the merge decision from recorded results without API calls."""
import json
from pathlib import Path
import statistics

DIRECTORY = Path(__file__).resolve().parent


def main():
    cases = []
    for label in ("titanic", "malala", "cctv"):
        directory = DIRECTORY / f"merge_final_{label}_20261020"
        data = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
        result = {"case": label, "repetitions_per_layout": len(data["runs"]) // 2,
                  "requests": sum(r["usage"]["requests"] for r in data["runs"]),
                  "failed_requests": sum(r["usage"]["failed_requests"] for r in data["runs"])}
        for variant in ("original", "revised"):
            runs = [r for r in data["runs"] if r["variant"] == variant]
            result[variant] = {
                "synopsis_median_sec": statistics.median(r["synopsis_wall_sec"] for r in runs),
                "sampled_scene_median_sec": statistics.median(r["sampled_scenes_wall_sec"] for r in runs),
                "combined_median_sec": statistics.median(r["synopsis_wall_sec"] + r["sampled_scenes_wall_sec"] for r in runs),
                "required_question_order_passes": sum(r["quality"]["required_questions_in_order"] for r in runs),
                "mean_missing_answer_count": statistics.mean(r["quality"]["missing_answer_count"] for r in runs),
            }
        cases.append(result)
    smoke_directory = DIRECTORY / "merge_release_final_20261020"
    smoke = json.loads((smoke_directory / "metrics.json").read_text(encoding="utf-8"))
    output = json.loads((smoke_directory / "revised_1/synopsis.json").read_text(encoding="utf-8"))
    meaningful = lambda text: bool(str(text).strip()) and str(text).strip().lower().rstrip(".") != "not explicitly stated"
    check = dict(smoke["runs"][0])
    check["meaningful_highlights"] = sum(meaningful(item["highlight"]) for item in output["video_highlights"])
    check["meaningful_timeline_events"] = sum(meaningful(item["event"]) for item in output["video_timeline"])
    check["meaningful_clips"] = sum(meaningful(item["description"]) for item in output["suggested_clips"])
    check["experimental_cache_layout_enabled"] = smoke["experimental_cache_layout_enabled"]
    if (any(check[key] == 0 for key in ("meaningful_highlights", "meaningful_timeline_events", "meaningful_clips"))
            or check["quality"]["fully_missing_required_answers"] or check["experimental_cache_layout_enabled"]
            or check["usage"]["failed_requests"]):
        raise ValueError("Final default smoke does not meet the release checks")
    report = {"decision": "Prepare default-layout quality fixes and telemetry for merge; experimental rearrangement stays disabled.",
              "experimental_layout_comparisons": cases, "release_smoke": check,
              "quality_review": "Local source-based spot review of summary, five required answers, and two highlights per layout for each of three videos. Not a complete factual audit.",
              "external_judge": "Not run: automatic approval review rejected additional export of saved narratives, evidence and outputs. Local review used instead."}
    (DIRECTORY / "MERGE_METRICS.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# Merge readiness", "", report["decision"], "",
             "## Performance decision", "",
             "The experimental shared-narrative ordering did not satisfy the goal of equal or better speed.",
             "Both layouts received the same answer-matching, separator, and timestamp safeguards in these comparisons.",
             "Runs alternated order. These are saved-input LLM stages, not whole-video processing.", "",
             "| Input | Runs per layout | Original synopsis median | Experimental synopsis median | Original scene sample median | Experimental scene sample median |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for case in cases:
        old, new = case["original"], case["revised"]
        lines.append(f"| {case['case']} | {case['repetitions_per_layout']} | {old['synopsis_median_sec']:.2f}s | {new['synopsis_median_sec']:.2f}s | {old['sampled_scene_median_sec']:.2f}s | {new['sampled_scene_median_sec']:.2f}s |")
    lines += ["", f"Total comparison requests: {sum(c['requests'] for c in cases)}; failures: {sum(c['failed_requests'] for c in cases)}.",
              "Scene timings were mixed; do not extrapolate three-scene samples to a complete video pipeline.",
              "Cold-cache equivalence was not established. Repeated original prompts also cache automatically.",
              "", "## Merge defaults", "",
              "- Original task-first prompt order is the default. The runtime layout helper returns the prompt unchanged.",
              "- Provider-side caching remains available with the existing API settings; telemetry reports actual cache usage.",
              "- Experimental rearrangement requires `KAIROS_CACHE_FRIENDLY_PROMPTS=1`; it is not enabled by default.",
              "- All seven primary synopsis prompts are byte-identical to the quality-fixed baseline in the integration fixture.",
              "- The same mocked model responses produce identical complete synopsis objects for both default paths.",
              "- No model, generation setting, concurrency, or normal retry policy changes were introduced.",
              "", "## Quality and release smoke", "",
              "Question matching tolerates punctuation differences without assigning answers by position.",
              "The timestamp guard rejects unsupported values, preserves explicitly timed scene headings, and retains event descriptions.",
              "Missing event times remain unknown; the guard does not recover lost timing or prove semantic alignment.", "",
              f"Final default smoke: {check['synopsis_wall_sec']:.2f}s, {check['usage']['successful_requests']} successful calls, {check['quality']['question_count']} questions, {check['quality']['missing_answer_count']} missing-answer placeholders.",
              f"Meaningful event descriptions: {check['meaningful_highlights']} highlights, {check['meaningful_timeline_events']} timeline events, {check['meaningful_clips']} clips.",
              "This single smoke run is a correctness check, not evidence of a speedup.", "",
              "Local review found the sampled main summaries and required answers consistent with their saved narratives.",
              "The review also exposed blank highlight descriptions in one baseline run; final prompts now explicitly preserve events when timing is missing.",
              "The external judge was not run after automatic approval review rejected the additional data export.",
              "There is no claim of full factual equivalence for every stochastic output.", "",
              "## Validation", "",
              "28 focused tests pass: prompt rendering and both layouts, answer matching, timestamp grounding, concurrent logging, unknown usage, API-error propagation, repairs, file output, scene-history ordering, custom-template preservation, and baseline request/output parity, including all three scene template API payloads.",
              "Use the focused suite documented in README.md; unrelated video-model and live-connection tests are not part of this change.",
              "`git diff --check` passes. Changes are prepared locally; no push or merge has been performed."]
    (DIRECTORY / "MERGE_READINESS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"comparison_requests": sum(c['requests'] for c in cases), "release_smoke": check}, indent=2))


if __name__ == "__main__":
    main()
