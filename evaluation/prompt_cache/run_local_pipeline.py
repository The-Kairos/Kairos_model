"""Run every LLM stage locally from saved non-LLM scene evidence."""
import argparse
import copy
from datetime import datetime, timezone
import functools
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.path_utils import load_kairos_env
from run_benchmark import aggregate, quality

CASES = {
    "speech": "Watch Malala Yousafzai's Nobel Peace Prize acceptance speech.mp4_20260319_082306.json",
    "story": "Young Sheldon - First Day of High School.mp4_20260329_115239.json",
    "visual": "CCTV Dogs.mp4_20260312_213941.json",
}


def read_evidence(path):
    data = json.loads(path.read_bytes())
    scenes = [{key: copy.deepcopy(value) for key, value in scene.items()
               if not key.startswith("llm_")} for scene in data["scenes"]]
    return {"video_path": data.get("video_path", path.stem), "scenes": scenes}


def load_synopsis(name):
    path = ROOT / "src/synopsis_synthesis.py"
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), module.__dict__)
    return module


def run_stages(scene_module, synopsis_module, evidence, client, config, output, progress=print):
    """Actual production stage functions; no reuse of saved LLM outputs."""
    walls = {}
    started = time.perf_counter()
    progress("scene descriptions", flush=True)
    stage_started = time.perf_counter()
    described = scene_module.describe_scenes(
        copy.deepcopy(evidence["scenes"]), client, hist_size=config["scene_history"],
        ASR_key="audio_speech", AST_key="audio_natural", model="gpt-4o",
        max_workers=config["scene_workers"], cooldown_sec=config["scene_cooldown_sec"],
        video_path=evidence["video_path"])
    walls["scene_descriptions"] = time.perf_counter() - stage_started
    if len(described) != len(evidence["scenes"]):
        raise RuntimeError("Scene generation dropped scenes; this is not a valid speed comparison")
    progress("narrative generation", flush=True)
    stage_started = time.perf_counter()
    narrative = synopsis_module.summarize_scenes(
        client, config["deployment"], described, chunk_size=config["chunk_chars"],
        summary_len=config["target_narrative_chars"], max_workers=config["narrative_workers"],
        reduce_group_size=config["reduce_group_size"])
    walls["narrative"] = time.perf_counter() - stage_started
    narrative["video_path"] = evidence["video_path"]
    progress("final report", flush=True)
    stage_started = time.perf_counter()
    result = synopsis_module.synthesize_synopsis(
        client, config["deployment"], narrative, output_dir=str(output), consistency_pass_mode="off")
    walls["synopsis"] = time.perf_counter() - stage_started
    walls["total_llm"] = time.perf_counter() - started
    return result, walls


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "evaluation/prompt_cache/local_pipeline_20261020")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--cases", nargs="+", choices=tuple(CASES), default=list(CASES))
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("repetitions must be positive")
    os.chdir(ROOT)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "metrics.json").exists():
        parser.error("Use a fresh output directory; existing measurements are preserved")
    load_kairos_env()
    from openai import AzureOpenAI
    import psutil
    from src import scene_description as scene_module
    client = AzureOpenAI(api_key=os.environ["GPT_KEY"], azure_endpoint=os.environ["GPT_ENDPOINT"],
                         api_version=os.environ["GPT_VERSION"], max_retries=0, timeout=120)
    config = dict(deployment=os.environ["GPT_DEPLOYMENT"], scene_history=5, scene_workers=8,
                  narrative_workers=6, chunk_chars=20000, target_narrative_chars=50000,
                  reduce_group_size=4, scene_cooldown_sec=0, temperature_scene=0.3,
                  temperature_summary=0.2, synopsis_concurrency=7, sdk_retries=0,
                  api_version=os.environ["GPT_VERSION"])
    sources = {name: ROOT / "logs/_processed" / CASES[name] for name in args.cases}
    evidence = {name: read_evidence(path) for name, path in sources.items()}
    report = {
        "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
        "host": dict(platform.uname()._asdict(), python=sys.version, executable=sys.executable,
                     logical_cpus=os.cpu_count(), total_ram_bytes=psutil.virtual_memory().total),
        "packages": {name: version(name) for name in ("openai", "httpx", "python-dotenv", "psutil")},
        "config": config, "endpoint_sha256": hashlib.sha256(os.environ["GPT_ENDPOINT"].encode()).hexdigest(),
        "sources": {name: {"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                           "scene_count": len(evidence[name]["scenes"]),
                           "evidence_sha256": hashlib.sha256(json.dumps(evidence[name], sort_keys=True).encode()).hexdigest()}
                    for name, path in sources.items()},
        "working_files_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in [ROOT / "src/synopsis_synthesis.py", ROOT / "src/scene_description.py",
                                              ROOT / "src/prompt_layout.py", ROOT / "src/track_formatting.py",
                                              *sorted((ROOT / "prompts").glob("*.txt"))]},
        "scope": "All saved scenes, production two-pass scene history, newly generated narrative, newly generated synopsis.",
        "excluded": ["video decoding", "scene detection", "ASR inference", "caption/detection inference", "embeddings"],
        "limitations": ["Azure model inference and network latency are included in locally measured wall times.",
                        "First trials are not verified cold; no cache clearing or artificial cache-busting is used.",
                        "Repeats regenerate intermediate context; they are not necessarily exact-prompt cache repeats.",
                        "Tree reduction runs only if the production 50000-character narrative threshold is exceeded.",
                        "SDK retries are disabled for visibility; normal application retries remain enabled.",
                        "Format and coverage checks are not exhaustive factual verification."], "runs": []}
    metrics_path = args.output / "requests.jsonl"
    os.environ["KAIROS_LLM_METRICS_PATH"] = str(metrics_path)
    captured = []
    original_scene_call = scene_module.describe_flash_scene
    @functools.wraps(original_scene_call)
    def scene_call(text, client, **kwargs):
        kwargs["gpt_deployment"] = config["deployment"]
        result = original_scene_call(text, client, **kwargs)
        captured.append({"stage": "scene:" + Path(kwargs.get("prompt_path", "")).stem,
                         "input_sha256": hashlib.sha256(text.encode()).hexdigest(), "text": result})
        return result
    scene_module.describe_flash_scene = scene_call

    def save():
        (args.output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    save()
    for repetition in range(1, args.repetitions + 1):
        for case in args.cases:
            order = ("original", "cache_friendly") if repetition % 2 else ("cache_friendly", "original")
            for variant in order:
                run_id = f"{case}_{variant}_{repetition}"
                os.environ["KAIROS_LLM_RUN_ID"] = run_id
                os.environ["KAIROS_CACHE_FRIENDLY_PROMPTS"] = "1" if variant == "cache_friendly" else "0"
                captured.clear()
                module = load_synopsis("pipeline_" + run_id)
                original_gpt = module.call_gpt
                def capture_gpt(*call_args, **kwargs):
                    result = original_gpt(*call_args, **kwargs)
                    captured.append({"stage": kwargs.get("stage", "summary"), "text": result})
                    return result
                module.call_gpt = capture_gpt
                destination = args.output / run_id
                destination.mkdir()
                print("START " + run_id, flush=True)
                record = dict(run_id=run_id, case=case, variant=variant, repetition=repetition,
                              trial_kind="first_use_not_verified_cold" if repetition == 1 else "repeat_video_regenerated_context",
                              available_ram_bytes=psutil.virtual_memory().available)
                try:
                    result, walls = run_stages(scene_module, module, evidence[case], client, config, destination,
                                              progress=lambda stage, **kw: print(run_id + ": " + stage, **kw))
                    rows = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
                    requests = [row for row in rows if row.get("run_id") == run_id and row.get("kind") == "request"]
                    record.update(status="complete", wall_sec=walls, usage=aggregate(requests),
                                  response_models=sorted({r["model"] for r in requests if r.get("model")}),
                                  quality=quality(result["synopsis"], module.REQUIRED_QUESTIONS),
                                  scenes_in=len(evidence[case]["scenes"]), scenes_out=len(result["scenes"]),
                                  narrative_chars=len(result["narratives"][-1]["narrative"]),
                                  narrative_passes=len(result["narratives"]),
                                  tree_reduce_requests=sum(r["stage"] == "narrative:reduce" for r in requests),
                                  additional_synopsis_requests=max(0, sum(r["stage"].startswith(("synopsis:", "synthesize_synopsis:")) for r in requests) - 7))
                    (destination / "pipeline_outputs.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
                    (destination / "raw_sections.json").write_text(json.dumps(captured, indent=2), encoding="utf-8")
                    print(f"DONE {run_id}: {walls['total_llm']:.2f}s", flush=True)
                except Exception as exc:
                    record.update(status="failed", error_type=type(exc).__name__)
                    print("FAILED " + run_id + ": " + type(exc).__name__, flush=True)
                report["runs"].append(record)
                save()
                if record["status"] != "complete":
                    report["status"] = "incomplete"
                    save()
                    return 1
    report["status"] = "complete"
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    save()
    print("COMPLETE " + str(args.output), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
