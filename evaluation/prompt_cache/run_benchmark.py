"""Compare committed original prompts with working-tree cache-friendly prompts.

Runs only saved-input LLM stages. Credentials are read locally and never written.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.path_utils import load_kairos_env
from src.llm_metrics import measured_chat_completion


def git_text(ref, path):
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={ROOT.as_posix()}", "show", f"{ref}:{path}"],
        cwd=ROOT).decode("utf-8")


def original_module(ref, quality_matched=False):
    source = git_text(ref, "src/synopsis_synthesis.py")
    source = source.replace('return (PROMPTS_DIR / filename).read_text(encoding="utf-8")',
                            'return _original_template(filename)')
    source = source.replace('response = client.chat.completions.create(',
                            'response = _measured(client, stage="summary", attempt=attempt + 1,')
    module = types.ModuleType("cache_baseline")
    module.__file__ = str(ROOT / "src/synopsis_synthesis.py")
    def template(name):
        if quality_matched:
            # Current task-first templates contain the shared quality wording.
            # Only the candidate loader applies the experimental rearrangement.
            return (ROOT / "prompts" / name).read_text(encoding="utf-8")
        return git_text(ref, "prompts/" + name)
    module._original_template = template
    module._measured = measured_chat_completion
    if quality_matched:
        from src import synopsis_synthesis as current
        module._ground_synopsis_timestamps = current._ground_synopsis_timestamps
        source = source.replace('    synopsis_text = json.dumps(synopsis_json, ensure_ascii=False)',
            '    synopsis_json = _ground_synopsis_timestamps(synopsis_json, narrative_text)\n'
            '    synopsis_text = json.dumps(synopsis_json, ensure_ascii=False)')
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    if quality_matched:
        module._normalize_question_key = current._normalize_question_key
    return module


def quality(synopsis, required):
    questions = synopsis.get("questions", [])
    missing = lambda value: not isinstance(value, str) or not value.strip() or value.strip().lower().rstrip(".") == "not explicitly stated"
    return {
        "summary_present": not missing(synopsis.get("summary")),
        "title_present": not missing(synopsis.get("chat_name")),
        "highlight_count": len(synopsis.get("video_highlights", [])),
        "timeline_count": len(synopsis.get("video_timeline", [])),
        "clip_count": len(synopsis.get("suggested_clips", [])),
        "question_count": len(questions),
        "required_questions_in_order": [q.get("question") for q in questions[:len(required)]] == required,
        "missing_answer_count": sum("not explicitly stated" in str(q.get("answer", "")).lower() for q in questions),
        "fully_missing_required_answers": sum(missing(q.get("answer")) for q in questions[:len(required)]),
        "placeholder_questions": sum(str(q.get("question", "")).startswith("Additional predicted question") for q in questions),
        "factual_accuracy": "requires source review; not established by format checks",
    }


def aggregate(rows):
    good = [r for r in rows if r.get("kind") == "request" and not r.get("error_type")]
    prompt = sum(r.get("prompt_tokens") or 0 for r in good)
    known = [r for r in good if r.get("cached_tokens") is not None]
    cached = sum(r["cached_tokens"] for r in known) if known else None
    known_prompt = sum(r.get("prompt_tokens") or 0 for r in known)
    return {
        "requests": len(rows), "successful_requests": len(good),
        "failed_requests": sum(bool(r.get("error_type")) for r in rows),
        "cache_usage_reported_requests": sum(r.get("cached_tokens") is not None for r in good),
        "cache_hit_requests": sum((r.get("cached_tokens") or 0) > 0 for r in good),
        "prompt_tokens": prompt, "cached_tokens": cached,
        "cached_token_fraction": cached / known_prompt if known_prompt and cached is not None else None,
        "completion_tokens": sum(r.get("completion_tokens") or 0 for r in good),
        "median_request_sec": statistics.median(r["elapsed_sec"] for r in good) if good else None,
        "retry_attempts": sum(r.get("attempt", 1) > 1 for r in rows),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "logs/_processed/Titanic.1997.mkv_20260312_201859.json")
    parser.add_argument("--output", type=Path, default=ROOT / "evaluation/prompt_cache/results")
    parser.add_argument("--baseline-ref", default="c32882928a4650f77717a9379acd9593a42c45bb")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--scene-samples", type=int, default=3)
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--variant", choices=("both", "original", "revised"), default="both")
    parser.add_argument("--match-quality-fixes", action="store_true",
                        help="Apply current question matching and timestamp safeguards to the original layout too")
    parser.add_argument("--experimental-layout", action="store_true", help="Opt in to the cache-friendly prompt rearrangement")
    args = parser.parse_args()
    if args.repetitions < 1 or args.scene_samples < 0:
        parser.error("repetitions must be positive and scene-samples nonnegative")
    args.output.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output.resolve() / "requests.jsonl"
    if metrics_path.exists():
        parser.error("Use a new output directory to preserve earlier measurements")
    load_kairos_env()
    from openai import AzureOpenAI
    os.environ["KAIROS_LLM_METRICS_PATH"] = str(metrics_path)
    os.environ["KAIROS_LLM_RUN_ID"] = "probe"
    report = {"started_utc": datetime.now(timezone.utc).isoformat(), "runs": [],
              "status": "starting", "limitations": [
                  "No forced cache clearing; first use does not establish a cold cache.",
                  "Saved-input LLM stages only; no full video processing speed claim.",
                  "Output formats are checked automatically; factual accuracy needs source review.",
                  "SDK retries disabled in this benchmark; pipeline retry behavior remains enabled."]}

    def save():
        (args.output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    missing = [k for k in ("GPT_KEY", "GPT_ENDPOINT", "GPT_VERSION", "GPT_DEPLOYMENT") if not os.getenv(k)]
    if missing:
        report.update(status="blocked_missing_configuration", missing_variables=missing)
        save()
        return 1
    client = AzureOpenAI(api_key=os.environ["GPT_KEY"], azure_endpoint=os.environ["GPT_ENDPOINT"],
                         api_version=os.environ["GPT_VERSION"], max_retries=0, timeout=120)
    deployment = os.environ["GPT_DEPLOYMENT"]
    try:
        probe = measured_chat_completion(client.with_options(timeout=15), stage="connectivity_probe",
            model=deployment, messages=[{"role": "user", "content": "Reply OK."}], max_tokens=2)
        report["response_model"] = probe.model
        print("Connectivity probe succeeded", flush=True)
    except Exception as exc:
        report.update(status="blocked_api_access", error_type=type(exc).__name__,
                      status_code=getattr(exc, "status_code", None))
        # Store only exception class and status; API messages can include sensitive URLs.
        cause = exc.__cause__
        report["cause_type"] = type(cause).__name__ if cause else None
        save()
        print(json.dumps({k: report[k] for k in ("status", "error_type", "status_code", "cause_type")}), flush=True)
        return 1
    if args.probe_only:
        report["status"] = "probe_only"
        save()
        return 0

    if args.experimental_layout:
        os.environ["KAIROS_CACHE_FRIENDLY_PROMPTS"] = "1"
    from src import synopsis_synthesis as revised
    from src.scene_description import describe_flash_scene
    ref = subprocess.check_output(["git", "-c", f"safe.directory={ROOT.as_posix()}", "rev-parse", args.baseline_ref], cwd=ROOT, text=True).strip()
    baseline = original_module(ref, quality_matched=args.match_quality_fixes)
    report["quality_fixes_matched"] = args.match_quality_fixes
    from src.prompt_layout import cache_layout_enabled
    report["experimental_cache_layout_enabled"] = cache_layout_enabled()
    captured = []
    def capture_calls(function):
        def wrapped(*call_args, **call_kwargs):
            text = function(*call_args, **call_kwargs)
            caller = inspect.currentframe().f_back
            captured.append({"run_id": os.environ.get("KAIROS_LLM_RUN_ID"),
                "stage": call_kwargs.get("stage") or caller.f_locals.get("name", caller.f_code.co_name),
                "text": text})
            return text
        return wrapped
    baseline.call_gpt = capture_calls(baseline.call_gpt)
    revised.call_gpt = capture_calls(revised.call_gpt)
    report["baseline_commit"] = ref
    report["working_files_sha256"] = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [ROOT / "src/synopsis_synthesis.py", ROOT / "src/scene_description.py", ROOT / "src/llm_metrics.py", *sorted((ROOT / "prompts").glob("*.txt"))]
    }
    raw = args.input.read_bytes()
    data = json.loads(raw)
    report.update(input_path=str(args.input), input_sha256=hashlib.sha256(raw).hexdigest(),
                  narrative_chars=len(data["narratives"][-1]["narrative"]), scene_samples=args.scene_samples)
    baseline_prompt_dir = args.output / "baseline_prompts"
    baseline_prompt_dir.mkdir(exist_ok=True)
    for name in ("describe_scene.txt", "describe_scene_short.txt", "fallback_describe_scene.txt"):
        (baseline_prompt_dir / name).write_text(git_text(ref, "prompts/" + name), encoding="utf-8")
    scenes = data.get("scenes", [])
    indices = sorted(set(round(i * (len(scenes) - 1) / max(1, args.scene_samples - 1)) for i in range(args.scene_samples))) if scenes else []
    # Serialize saved evidence directly; do not import local video models.
    # Both arms receive identical evidence. This samples standalone scene calls,
    # not the production two-stage scene-history construction.
    evidence_keys = ("start_timecode", "end_timecode", "frame_captions", "yolo_detections", "audio_speech", "audio_natural")
    sample_texts = [json.dumps({key: scenes[i].get(key) for key in evidence_keys}, ensure_ascii=False) for i in indices]
    report["limitations"].append("Scene sample uses saved evidence JSON without production two-stage scene history.")
    report["sample_scene_indices"] = indices
    report["status"] = "running"
    save()
    for repetition in range(args.repetitions):
        order = [("original", baseline), ("revised", revised)]
        if repetition % 2:
            order.reverse()
        if args.variant != "both":
            order = [(name, module) for name, module in order if name == args.variant]
        for variant, module in order:
            run_id = f"{variant}_{repetition + 1}"
            os.environ["KAIROS_LLM_RUN_ID"] = run_id
            print("Starting " + run_id, flush=True)
            started = time.perf_counter()
            result = module.synthesize_synopsis(client, deployment, data,
                output_dir=str(args.output / run_id), consistency_pass_mode="off")
            wall = time.perf_counter() - started
            (args.output / run_id / "raw_sections.json").write_text(
                json.dumps([item for item in captured if item["run_id"] == run_id], indent=2), encoding="utf-8")
            scene_started = time.perf_counter()
            prompt = baseline_prompt_dir / "describe_scene.txt" if variant == "original" else ROOT / "prompts/describe_scene.txt"
            with ThreadPoolExecutor(max_workers=8) as pool:
                futures = [pool.submit(describe_flash_scene, text, client, prompt_path=str(prompt),
                            model="gpt-4o", gpt_deployment=deployment, video_path=data.get("video_path"),
                            cache_friendly_layout=(variant == "revised" and cache_layout_enabled())) for text in sample_texts]
                scene_outputs = []
                for future in futures:
                    try:
                        scene_outputs.append({"text": future.result()})
                    except Exception as exc:
                        scene_outputs.append({"error_type": type(exc).__name__})
            scene_wall = time.perf_counter() - scene_started
            (args.output / run_id / "scene_outputs.json").write_text(json.dumps(scene_outputs, indent=2), encoding="utf-8")
            rows = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
            rows = [r for r in rows if r.get("run_id") == run_id and r.get("kind") == "request"]
            report["runs"].append(dict(run_id=run_id, variant=variant, repetition=repetition + 1,
                synopsis_wall_sec=wall, sampled_scenes_wall_sec=scene_wall,
                usage=aggregate(rows), quality=quality(result["synopsis"], module.REQUIRED_QUESTIONS)))
            save()
            print(f"Finished {run_id}: synopsis {wall:.2f}s; scenes {scene_wall:.2f}s", flush=True)
    report["status"] = "complete"
    report["median_synopsis_wall_sec"] = {v: statistics.median(r["synopsis_wall_sec"] for r in report["runs"] if r["variant"] == v) for v in {r["variant"] for r in report["runs"]}}
    old, new = (report["median_synopsis_wall_sec"].get(v) for v in ("original", "revised"))
    report["synopsis_wall_reduction_fraction"] = 1 - new / old if old and new is not None else None
    save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
