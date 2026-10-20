"""Publish aggregate findings without copying raw model outputs or traces."""
import argparse
import copy
import json
from pathlib import Path


def publish(source, destination):
    report = json.loads((source / "metrics.json").read_text(encoding="utf-8"))
    published = copy.deepcopy(report)
    published["host"].pop("executable", None)
    comparisons = []
    for case in report["sources"]:
        repetitions = sorted({r["repetition"] for r in report["runs"] if r["case"] == case})
        for repetition in repetitions:
            old = json.loads((source / f"{case}_original_{repetition}/pipeline_outputs.json").read_text(encoding="utf-8"))
            new = json.loads((source / f"{case}_cache_friendly_{repetition}/pipeline_outputs.json").read_text(encoding="utf-8"))
            comparisons.append({
                "case": case, "repetition": repetition,
                "identical_final_report": old["synopsis"] == new["synopsis"],
                "identical_final_narrative": old["narratives"][-1]["narrative"] == new["narratives"][-1]["narrative"],
                "identical_scene_reports": sum(a["llm_scene_description"] == b["llm_scene_description"] for a, b in zip(old["scenes"], new["scenes"])),
                "scene_reports_compared": len(old["scenes"]),
            })
    published["exact_output_comparison"] = comparisons
    published["publication_scope"] = "Aggregate metadata only; raw model outputs, source evidence, and request traces remain local."
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "LOCAL_PIPELINE_METRICS.json").write_text(json.dumps(published, indent=2), encoding="utf-8")
    documents = {"CONCLUSION.md": "LOCAL_PIPELINE_CONCLUSION.md", "RESULTS.md": "LOCAL_PIPELINE_RESULTS.md", "QUALITY_REVIEW.md": "LOCAL_PIPELINE_QUALITY.md"}
    for name, target in documents.items():
        text = (source / name).read_text(encoding="utf-8")
        for old_name, new_name in documents.items():
            text = text.replace("](" + old_name + ")", "](" + new_name + ")")
        text = text.replace("](metrics.json)", "](LOCAL_PIPELINE_METRICS.json)")
        text += "\nPublished summary only. Raw outputs, review bundles, and request traces referenced above remain in the local ignored run directory.\n"
        if name == "CONCLUSION.md":
            identical = sum(c["identical_final_report"] for c in comparisons)
            text += f"\nExact comparison: {identical} of {len(comparisons)} paired final reports were identical. Different wording alone is not a quality failure; the separate source review found substantive differences as well.\n"
        (destination / target).write_text(text, encoding="utf-8")
    print(json.dumps({"paired_reports": len(comparisons), "identical_final_reports": sum(c["identical_final_report"] for c in comparisons)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--destination", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    publish(args.source, args.destination)
