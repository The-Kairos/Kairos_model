"""Prepare label-hidden local review bundles; no external judge or API calls."""
import argparse
import json
from pathlib import Path
import random

from run_local_pipeline import ROOT, read_evidence


def build(directory):
    report = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    review = directory / "local_review"
    review.mkdir(exist_ok=True)
    mapping = {}
    rng = random.Random(694)
    for case, source in report["sources"].items():
        folders = {variant: directory / f"{case}_{variant}_1" for variant in ("original", "cache_friendly")}
        if any(not (folder / "pipeline_outputs.json").exists() for folder in folders.values()):
            continue
        source = read_evidence(ROOT / source["path"].replace("\\", "/"))
        indices = sorted({0, len(source["scenes"]) // 2, len(source["scenes"]) - 1})
        variants = list(folders)
        rng.shuffle(variants)
        mapping[case] = dict(zip(("A", "B"), variants))
        fields = ("scene_index", "start_timecode", "end_timecode", "frame_captions", "audio_speech", "audio_natural", "yolo_detections")
        bundle = {"method": "First-trial local spot review; version labels hidden, not an exhaustive factual audit.",
                  "source_video": source["video_path"],
                  "all_source_speech": [{"time": s.get("start_timecode"), "speech": s.get("audio_speech")} for s in source["scenes"]],
                  "selected_source_scenes": [{k: source["scenes"][i].get(k) for k in fields} for i in indices], "outputs": {}}
        for label, variant in mapping[case].items():
            result = json.loads((folders[variant] / "pipeline_outputs.json").read_text(encoding="utf-8"))
            synopsis = result["synopsis"]
            bundle["outputs"][label] = {
                "summary": synopsis["summary"], "highlights": synopsis["video_highlights"],
                "selected_questions": [synopsis["questions"][i] for i in (0, 1, 2, 5, 6, 8, 9, 10, 13)],
                "selected_scene_reports": [{"scene_index": result["scenes"][i]["scene_index"],
                                            "description": result["scenes"][i]["llm_scene_description"]} for i in indices]}
        (review / f"{case}.json").write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    (review / "mapping.json").write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    print("Local label-hidden bundles prepared for: " + ", ".join(mapping))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    build(parser.parse_args().directory)
