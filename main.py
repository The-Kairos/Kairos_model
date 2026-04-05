from src.path_utils import load_kairos_env

# Load environment variables from project root dynamically
load_kairos_env(override=True)

from src.debug_utils import *
from src.log_utils import *
from src.redo_utils import apply_redo, REDO_CHOICES
import argparse
import io
import json
import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

# --- Mute harmless native warnings (FFmpeg / H264 / OpenCV) ---
os.environ["OPENCV_LOG_LEVEL"] = "OFF"
os.environ["AV_LOG_LEVEL"] = "quiet"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*mmco: unref short failure.*")

import gc
import torch

from src.path_utils import is_low_mem
from src.storage_utils import StorageManager

# --- Resource Policy Detection ---
LOW_MEM_MODE = is_low_mem()


def purge_memory(force: bool = False):
    """Clear RAM and GPU VRAM if in low memory mode or forced."""
    if not LOW_MEM_MODE and not force:
        return
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


use_gemini = False
client = None
if use_gemini:
    api_key = os.getenv("GEMINI_API_KEY")
    from google import genai
    model_name = "gemini-2.5-flash"
    deployment = model_name
else:
    model_name = "gpt-4o"
    endpoint = os.getenv("GPT_ENDPOINT")
    deployment = os.getenv("GPT_DEPLOYMENT")
    subscription_key = os.getenv("GPT_KEY")
    api_version = os.getenv("GPT_VERSION")

    from openai import AzureOpenAI


def get_llm_client():
    global client
    if client is not None:
        return client
    if use_gemini:
        api_key = os.getenv("GEMINI_API_KEY")
        from google import genai

        client = genai.Client(vertexai=True, api_key=api_key)
        return client

    if not endpoint or not subscription_key:
        raise RuntimeError("Azure OpenAI credentials are required to run the LLM pipeline.")

    client = AzureOpenAI(
        api_version=api_version,
        azure_endpoint=endpoint,
        api_key=subscription_key,
    )
    return client


improve_motion_detection = True
pyscene_threshold = 27
pyscene_shortest = 2
frames_per_scene = 3
frame_resolution = 320
blip_start_prompt = "a video frame of"
blip_caption_len = 50
blip_min_length = 15
blip_num_beams = 1
blip_do_sample = True
blip_top_p = 0.85
blip_temperature = 0.65
blip_length_penalty = 1.0
blip_no_repeat_ngram_size = 3
blip_repetition_penalty = 1.1
yolo_action_fps = 4
yolo_conf_thres = 0.8
yolo_iou_thres = 0.5
ast_target_sr = 16000
asr_model_size = "medium"
asr_use_vad = True
asr_target_sr = 16000
llm_scene_history = 5
llm_chunk_len = 20000
llm_summary_len = 50000
llm_cooldown_sec = 0
rag_top_k_context = 10

improve_motion_detection = False
prioritize_speed = False
process_static_videos = False

if improve_motion_detection:
    pyscene_threshold = 15
    pyscene_shortest = 0.5
    frames_per_scene = 5
    yolo_action_fps = 8
if prioritize_speed:
    pyscene_threshold = 40
    frames_per_scene = 1
    llm_chunk_len = 500000
    llm_summary_len = 500000
if process_static_videos:
    pyscene_threshold = 3
    frames_per_scene = 1
    yolo_action_fps = 0.5

params = {
    "improve_motion_detection": improve_motion_detection,
    "prioritize_speed": prioritize_speed,
    "process_static_videos": process_static_videos,
    "pyscene_threshold": pyscene_threshold,
    "pyscene_shortest": pyscene_shortest,
    "frames_per_scene": frames_per_scene,
    "frame_resolution": frame_resolution,
    "blip_start_prompt": blip_start_prompt,
    "blip_caption_len": blip_caption_len,
    "blip_min_length": blip_min_length,
    "blip_num_beams": blip_num_beams,
    "blip_do_sample": blip_do_sample,
    "blip_top_p": blip_top_p,
    "blip_temperature": blip_temperature,
    "blip_length_penalty": blip_length_penalty,
    "blip_no_repeat_ngram_size": blip_no_repeat_ngram_size,
    "blip_repetition_penalty": blip_repetition_penalty,
    "yolo_conf_thres": yolo_conf_thres,
    "yolo_iou_thres": yolo_iou_thres,
    "ast_target_sr": ast_target_sr,
    "asr_model_size": asr_model_size,
    "asr_use_vad": asr_use_vad,
    "asr_target_sr": asr_target_sr,
    "llm_scene_history": llm_scene_history,
    "llm_chunk_len": llm_chunk_len,
    "llm_summary_len": llm_summary_len,
    "llm_cooldown_sec": llm_cooldown_sec,
    "rag_top_k_context": rag_top_k_context,
}

STEP_STAGE_MAP = {
    "get_scene_list": ("scene_detection", 10),
    "save_clips": ("clip_extraction", 20),
    "sample_frames": ("frame_sampling", 30),
    "caption_frames": ("frame_captioning", 40),
    "sample_fps": ("motion_sampling", 45),
    "detect_object_yolo": ("object_detection", 50),
    "audio_scan": ("audio_prescan", 60),
    "asr_timings": ("speech_transcription", 65),
    "ast_timings": ("sound_analysis", 75),
    "describe_scenes": ("scene_description", 85),
    "summarize_scenes": ("narrative_synthesis", 90),
    "synthesize_synopsis": ("synopsis_generation", 95),
    "make_embedding": ("embedding", 100),
}

BENCHMARK_STEPS = [
    "get_scene_list",
    "save_clips",
    "sample_frames",
    "caption_frames",
    "sample_fps",
    "detect_object_yolo",
    "audio_scan",
    "asr_timings",
    "ast_timings",
    "describe_scenes",
    "summarize_scenes",
    "synthesize_synopsis",
    "make_embedding",
]


@contextmanager
def maybe_silence(enabled: bool):
    if not enabled:
        yield
        return
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        yield


def emit(message: str = "", quiet: bool = False, force: bool = False):
    if quiet and not force:
        return
    print(message)


def section(title: str, quiet: bool = False):
    if quiet:
        return
    print_section(title)


def all_scenes_have_key(scenes: list, key: str) -> bool:
    return bool(scenes) and all(isinstance(scene, dict) and key in scene for scene in scenes)


def clone_scenes(scenes: list) -> list:
    return [dict(scene) for scene in scenes]


def merge_scene_variants(base_scenes: list, *variants: list) -> list:
    merged = [dict(scene) for scene in base_scenes]
    index_map = {}
    for pos, scene in enumerate(merged):
        index_map[scene.get("scene_index", pos)] = pos

    for variant in variants:
        if not variant:
            continue
        for fallback_pos, scene in enumerate(variant):
            idx = scene.get("scene_index", fallback_pos)
            pos = index_map.get(idx)
            if pos is None:
                merged.append(dict(scene))
                index_map[idx] = len(merged) - 1
            else:
                merged[pos].update(scene)

    merged.sort(key=lambda item: item.get("scene_index", 0))
    return merged


def resolve_bool_env(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def resolve_embedding_config(provider_arg: str | None, model_arg: str | None) -> tuple[str, str]:
    provider = (provider_arg or os.environ.get("KAIROS_EMBEDDING_PROVIDER") or "gemini").strip().lower()
    if provider in {"openai", "azure", "azure_openai"}:
        provider = "openai"
        model = (
            model_arg
            or os.environ.get("KAIROS_EMBEDDING_MODEL")
            or os.environ.get("OPENAI_EMBEDDING_DEPLOYMENT")
            or os.environ.get("GPT_EMBEDDING_DEPLOYMENT")
            or "text-embedding-3-large"
        )
    else:
        provider = "gemini"
        model = model_arg or os.environ.get("KAIROS_EMBEDDING_MODEL") or "gemini-embedding-001"
    return provider, model


def get_ast_worker_count() -> int:
    raw = os.environ.get("KAIROS_AST_WORKERS") or os.environ.get("MAX_KAIROS_WORKERS")
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    return 2


def update_state_for_step(storage_manager: StorageManager, step_name: str):
    info = STEP_STAGE_MAP.get(step_name)
    if not info:
        return
    stage, percent = info
    storage_manager.update_pipeline_state(stage, percent)


def persist_step_updates(checkpoint: dict, step_store: dict, updates: dict, storage_manager: StorageManager):
    for step_name, step_log in updates.items():
        step_store[step_name] = step_log
        checkpoint.setdefault("steps", {})[step_name] = step_log
        update_state_for_step(storage_manager, step_name)


def benchmark_step_value(steps: dict, name: str) -> str:
    value = steps.get(name, {}).get("wall_time_sec")
    if value is None:
        return "-"
    return f"{float(value):.3f}"


def ensure_benchmark_plan(path: Path):
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# Parallelization Benchmark Plan\n\n"
        "This report compares Kairos production runs in two modes only:\n\n"
        "- `semi_parallel`: current production orchestration with internal parallelism in audio and LLM stages.\n"
        "- `parallel`: updated branch-parallel orchestration after scene detection.\n\n"
        "Each benchmark row captures wall time, embedding provider/model, and stage-level timings so we can compare architectural changes without maintaining a separate sequential pipeline.\n",
        encoding="utf-8",
    )


def append_benchmark_report(path: Path, entry: dict, steps: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(
            "# Parallelization Benchmarks\n\n"
            "| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |\n"
            "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |\n",
            encoding="utf-8",
        )

    row = (
        f"| {entry['timestamp']} | {entry['video_name']} | {entry['execution_mode']} | "
        f"{entry['embedding_provider']} | {entry['embedding_model']} | {entry['total_wall_time_sec']:.3f} | "
        f"{benchmark_step_value(steps, 'describe_scenes')} | {benchmark_step_value(steps, 'summarize_scenes')} | "
        f"{benchmark_step_value(steps, 'synthesize_synopsis')} | {benchmark_step_value(steps, 'make_embedding')} |\n"
    )

    detail_lines = [
        "",
        f"## {entry['timestamp']} | {entry['video_name']} | {entry['execution_mode']}",
        "",
        f"- Video path: `{entry['video_path']}`",
        f"- Low memory mode: `{entry['low_mem_mode']}`",
        f"- Debug: `{entry['debug']}`",
        f"- Quiet: `{entry['quiet']}`",
        f"- Embedding provider: `{entry['embedding_provider']}`",
        f"- Embedding model: `{entry['embedding_model']}`",
        f"- Total wall time: `{entry['total_wall_time_sec']:.3f}` sec",
        "",
        "| Step | Wall Time (sec) |",
        "| --- | ---: |",
    ]
    for step_name in BENCHMARK_STEPS:
        detail_lines.append(f"| {step_name} | {benchmark_step_value(steps, step_name)} |")

    with open(path, "a", encoding="utf-8") as handle:
        handle.write(row)
        handle.write("\n".join(detail_lines) + "\n")


def run_blip_branch(test_video: str, output_dir: str, scenes: list, debug: bool, quiet: bool):
    branch_scenes = clone_scenes(scenes)
    updates = {}
    frames_output_dir = f"{output_dir}/.frames" if debug else None

    if all_scenes_have_key(branch_scenes, "frame_captions"):
        return branch_scenes, updates

    if frames_output_dir:
        emit(f"Saving sampled frames in: {frames_output_dir}", quiet=quiet)
    with maybe_silence(quiet):
        branch_scenes, updates["sample_frames"] = sample_frames_log(
            input_video_path=test_video,
            scenes=branch_scenes,
            num_frames=frames_per_scene,
            new_size=frame_resolution,
            output_dir=frames_output_dir,
        )

    section("Running BLIP...", quiet=quiet)
    with maybe_silence(quiet):
        branch_scenes, updates["caption_frames"] = caption_frames_log(
            scenes=branch_scenes,
            prompt=blip_start_prompt,
            max_length=blip_caption_len,
            min_length=blip_min_length,
            num_beams=blip_num_beams,
            do_sample=blip_do_sample,
            top_p=blip_top_p,
            temperature=blip_temperature,
            length_penalty=blip_length_penalty,
            no_repeat_ngram_size=blip_no_repeat_ngram_size,
            repetition_penalty=blip_repetition_penalty,
            debug=debug,
        )
    return branch_scenes, updates


def run_yolo_branch(test_video: str, output_dir: str, scenes: list, debug: bool, quiet: bool):
    branch_scenes = clone_scenes(scenes)
    updates = {}
    fps_output_dir = f"{output_dir}/.fps" if debug else None
    yolo_output_dir = f"{output_dir}/.yolo" if debug else None

    if all_scenes_have_key(branch_scenes, "yolo_detections"):
        return branch_scenes, updates

    if not all_scenes_have_key(branch_scenes, "yolo_frames"):
        if fps_output_dir:
            emit(f"Saving sampled fps in: {fps_output_dir}", quiet=quiet)
        with maybe_silence(quiet):
            branch_scenes, updates["sample_fps"] = sample_fps_log(
                input_video_path=test_video,
                scenes=branch_scenes,
                fps=yolo_action_fps,
                new_size=frame_resolution,
                output_dir=fps_output_dir,
                frames_key="yolo_frames",
                frame_paths_key="yolo_frame_paths",
            )

    section("Running YOLOv8...", quiet=quiet)
    with maybe_silence(quiet):
        branch_scenes, updates["detect_object_yolo"] = detect_object_yolo_log(
            scenes=branch_scenes,
            model_size="model/yolov8s.pt",
            conf=yolo_conf_thres,
            iou=yolo_iou_thres,
            output_dir=yolo_output_dir,
            frame_key="yolo_frames",
            summary_key="yolo_detections",
            debug=debug,
        )
    return branch_scenes, updates


def run_audio_branch(
    test_video: str,
    scenes: list,
    debug: bool,
    quiet: bool,
    execution_mode: str,
):
    branch_scenes = clone_scenes(scenes)
    updates = {}
    scan_result = None

    needs_asr = not all_scenes_have_key(branch_scenes, "audio_speech")
    needs_ast = not all_scenes_have_key(branch_scenes, "audio_natural")
    if not needs_asr and not needs_ast:
        return branch_scenes, updates

    section("Running Audio Pre-Scan...", quiet=quiet)
    with maybe_silence(quiet):
        scan_result, updates["audio_scan"] = scan_audio_log(
            video_path=test_video,
            scenes=branch_scenes,
            target_sr=asr_target_sr,
            debug=debug,
        )

    ast_workers = get_ast_worker_count()

    def _run_asr(local_scenes):
        section("Running Whisper (Parallel)...", quiet=quiet)
        with maybe_silence(quiet):
            return extract_speech_log(
                scenes=local_scenes,
                scan_result=scan_result,
                model_size=asr_model_size,
                use_vad=asr_use_vad,
                language=None,
                parallel=True,
                use_api=True,
                force_cpu=True,
                debug=debug,
            )

    def _run_ast(local_scenes):
        section("Running MIT AST (Parallel)...", quiet=quiet)
        with maybe_silence(quiet):
            return extract_sounds_log(
                scenes=local_scenes,
                scan_result=scan_result,
                max_workers=ast_workers,
                use_processes=True,
                force_cpu=False,
                debug=debug,
            )

    if execution_mode == "parallel" and needs_asr and needs_ast:
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_asr = executor.submit(_run_asr, clone_scenes(branch_scenes))
            future_ast = executor.submit(_run_ast, clone_scenes(branch_scenes))
            asr_scenes, asr_log = future_asr.result()
            ast_scenes, ast_log = future_ast.result()
        branch_scenes = merge_scene_variants(branch_scenes, asr_scenes, ast_scenes)
        updates["asr_timings"] = asr_log
        updates["ast_timings"] = ast_log
    else:
        if needs_asr:
            branch_scenes, updates["asr_timings"] = _run_asr(branch_scenes)
        if needs_ast:
            branch_scenes, updates["ast_timings"] = _run_ast(branch_scenes)

    return branch_scenes, updates


def ensure_scene_detection(
    checkpoint: dict,
    step_store: dict,
    storage_manager: StorageManager,
    test_video: str,
    output_dir: str,
    debug: bool,
    quiet: bool,
):
    if checkpoint.get("scenes"):
        return checkpoint

    section("Running PysceneDetect...", quiet=quiet)
    with maybe_silence(quiet):
        checkpoint["scenes"], step_store["get_scene_list"] = get_scene_list_log(
            input_video_path=test_video,
            threshold=pyscene_threshold,
            min_scene_sec=pyscene_shortest,
        )
    update_state_for_step(storage_manager, "get_scene_list")
    if debug and not quiet:
        see_scenes_cuts(df=checkpoint["scenes"])

    if debug:
        emit(f"Saving clips in: {output_dir}/.clips", quiet=quiet)
        with maybe_silence(quiet):
            checkpoint["scenes"], step_store["save_clips"] = save_clips_log(
                video_path=test_video,
                scenes=checkpoint["scenes"],
                output_dir=f"{output_dir}/.clips",
            )
        update_state_for_step(storage_manager, "save_clips")
    storage_manager.save_checkpoint(checkpoint)
    purge_memory()
    return checkpoint


def run_visual_audio_pipeline(
    checkpoint: dict,
    step_store: dict,
    storage_manager: StorageManager,
    test_video: str,
    output_dir: str,
    execution_mode: str,
    debug: bool,
    quiet: bool,
):
    scenes = checkpoint.get("scenes", [])
    if not scenes:
        return checkpoint

    if execution_mode == "parallel":
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_blip = executor.submit(run_blip_branch, test_video, output_dir, scenes, debug, quiet)
            future_yolo = executor.submit(run_yolo_branch, test_video, output_dir, scenes, debug, quiet)
            future_audio = executor.submit(run_audio_branch, test_video, scenes, debug, quiet, execution_mode)

            blip_scenes, blip_updates = future_blip.result()
            yolo_scenes, yolo_updates = future_yolo.result()
            audio_scenes, audio_updates = future_audio.result()

        checkpoint["scenes"] = merge_scene_variants(scenes, blip_scenes, yolo_scenes, audio_scenes)
        persist_step_updates(checkpoint, step_store, blip_updates, storage_manager)
        persist_step_updates(checkpoint, step_store, yolo_updates, storage_manager)
        persist_step_updates(checkpoint, step_store, audio_updates, storage_manager)
        storage_manager.save_checkpoint(checkpoint)
        purge_memory()
        return checkpoint

    blip_scenes, blip_updates = run_blip_branch(test_video, output_dir, scenes, debug, quiet)
    checkpoint["scenes"] = merge_scene_variants(checkpoint["scenes"], blip_scenes)
    persist_step_updates(checkpoint, step_store, blip_updates, storage_manager)
    if blip_updates:
        storage_manager.save_checkpoint(checkpoint)
        purge_memory()

    yolo_scenes, yolo_updates = run_yolo_branch(test_video, output_dir, checkpoint["scenes"], debug, quiet)
    checkpoint["scenes"] = merge_scene_variants(checkpoint["scenes"], yolo_scenes)
    persist_step_updates(checkpoint, step_store, yolo_updates, storage_manager)
    if yolo_updates:
        storage_manager.save_checkpoint(checkpoint)
        purge_memory()

    audio_scenes, audio_updates = run_audio_branch(test_video, checkpoint["scenes"], debug, quiet, execution_mode)
    checkpoint["scenes"] = merge_scene_variants(checkpoint["scenes"], audio_scenes)
    persist_step_updates(checkpoint, step_store, audio_updates, storage_manager)
    if audio_updates:
        storage_manager.save_checkpoint(checkpoint)
        purge_memory()

    return checkpoint


def run_llm_and_rag_pipeline(
    checkpoint: dict,
    step_store: dict,
    storage_manager: StorageManager,
    test_video: str,
    output_dir: str,
    debug: bool,
    quiet: bool,
    embedding_provider: str,
    embedding_model: str,
):
    llm_client = get_llm_client()
    if checkpoint.get("scenes") and not all_scenes_have_key(checkpoint["scenes"], "llm_scene_description"):
        section("Running GPT4o Scene Descriptions...", quiet=quiet)
        with maybe_silence(quiet):
            checkpoint["scenes"], step_store["describe_scenes"] = describe_scenes_log(
                scenes=checkpoint["scenes"],
                client=llm_client,
                hist_size=llm_scene_history,
                YOLO_key="yolo_detections",
                FLIP_key="frame_captions",
                ASR_key="audio_speech",
                AST_key="audio_natural",
                SUMMARY_key="llm_scene_description",
                model=model_name,
                prompt_path="prompts/describe_scene.txt",
                cooldown_sec=llm_cooldown_sec,
                debug=debug,
                video_path=test_video,
            )
        update_state_for_step(storage_manager, "describe_scenes")
        storage_manager.save_checkpoint(checkpoint)
        purge_memory()

    if "narratives" not in checkpoint:
        section("Running GPT4o Summary narrative...", quiet=quiet)
        with maybe_silence(quiet):
            checkpoint, step_store["summarize_scenes"] = summarize_scenes_log(
                client=llm_client,
                deployment=deployment,
                scenes=checkpoint["scenes"],
                chunk_size=llm_chunk_len,
                summary_len=llm_summary_len,
                debug=debug,
                output_dir=output_dir,
            )
        update_state_for_step(storage_manager, "summarize_scenes")
        storage_manager.save_checkpoint(checkpoint)

    if "synopsis" not in checkpoint:
        section("Running GPT4o Synopsis generation...", quiet=quiet)
        with maybe_silence(quiet):
            checkpoint, step_store["synthesize_synopsis"] = synthesize_synopsis_log(
                client=llm_client,
                deployment=deployment,
                data=checkpoint,
                debug=debug,
                output_dir=output_dir,
            )
        update_state_for_step(storage_manager, "synthesize_synopsis")
        storage_manager.save_checkpoint(checkpoint)

    rag_path = Path(output_dir) / "rag_embedding.json"
    reproduce_embedding = False
    if not rag_path.exists():
        reproduce_embedding = True
    else:
        if "rag_embedding" not in checkpoint:
            try:
                with open(rag_path, "r", encoding="utf-8") as handle:
                    data = json.load(handle)
                    if isinstance(data, dict) and "embeddings" in data:
                        checkpoint["rag_embedding"] = data
                    else:
                        reproduce_embedding = True
            except (json.JSONDecodeError, OSError, Exception):
                reproduce_embedding = True

    if reproduce_embedding:
        with maybe_silence(quiet):
            checkpoint["rag_embedding"], step_store["make_embedding"] = make_embedding_log(
                checkpoint=checkpoint,
                output_path=str(rag_path),
                provider=embedding_provider,
                model=embedding_model,
            )
        update_state_for_step(storage_manager, "make_embedding")

    if storage_manager.is_remote:
        full_rag_data = None
        if rag_path.exists():
            try:
                with open(rag_path, "r", encoding="utf-8") as handle:
                    full_rag_data = json.load(handle)
            except Exception:
                full_rag_data = checkpoint.get("rag_embedding")

        storage_manager.save_final_results(
            checkpoint=checkpoint,
            rag_embedding=full_rag_data,
        )

    return checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Process videos or run RAG.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    process = subparsers.add_parser("process", help="Process videos")
    process.add_argument("path", nargs="?", help="Direct path to a video file for easy one-off runs")
    process.add_argument("--video", action="append", help="Blob name or path (repeatable)")
    process.add_argument("--all", action="store_true", help="Process all catalog videos")
    process.add_argument(
        "--filter",
        choices=["short", "medium", "long", "extra"],
        help="Inclusive length filter",
    )
    process.add_argument(
        "--include-unknown",
        action="store_true",
        help="Include videos with unknown length when filtering",
    )
    process.add_argument(
        "--redo",
        nargs="+",
        action="append",
        choices=REDO_CHOICES,
        help="Redo a processing step; dependents are redone by default (repeatable).",
    )
    process.add_argument(
        "--redo-only",
        nargs="*",
        choices=REDO_CHOICES,
        help=(
            "Redo only the specified steps and stop afterward (no dependents). "
            "Provide steps here or use with --redo."
        ),
    )
    process.add_argument("--chat-id", help="MongoDB Chat ID to update")
    process.add_argument("--mongo-uri", help="MongoDB Connection URI")
    process.add_argument(
        "--execution-mode",
        choices=["semi_parallel", "parallel"],
        default=os.environ.get("KAIROS_EXECUTION_MODE", "semi_parallel"),
        help="Choose the production orchestration mode.",
    )
    process.add_argument("--debug", action="store_true", help="Enable verbose debug output.")
    process.add_argument("--quiet", action="store_true", help="Reduce console output from the pipeline.")
    process.add_argument(
        "--embedding-provider",
        choices=["gemini", "openai"],
        default=os.environ.get("KAIROS_EMBEDDING_PROVIDER", "gemini"),
        help="Embedding provider for RAG generation.",
    )
    process.add_argument(
        "--embedding-model",
        help="Embedding model or deployment name to use for RAG generation.",
    )

    rag = subparsers.add_parser("rag", help="Run RAG for a single video")
    rag.add_argument("--video", required=True, help="Blob name or path")

    return parser.parse_args()


def main():
    videos_dir = Path("Videos")
    catalog_path = videos_dir / "_all_videos.json"
    processed_root = Path("_processed")
    args = parse_args()

    mongo_uri = args.mongo_uri or os.getenv("MONGODB_URI")
    storage_manager = StorageManager(
        chat_id=getattr(args, "chat_id", None),
        mongo_uri=mongo_uri,
        local_path=None,
    )

    redo_only_raw = getattr(args, "redo_only", None)
    redo_only_flag = redo_only_raw is not None
    redo_only_steps = redo_only_raw or []
    redo_steps = []

    def _flatten(values):
        flat = []
        if not values:
            return flat
        for value in values:
            if isinstance(value, (list, tuple)):
                flat.extend(value)
            else:
                flat.append(value)
        return flat

    if redo_only_steps:
        redo_steps = redo_only_steps
    elif getattr(args, "redo", None):
        redo_steps = _flatten(args.redo)
    if redo_only_flag and not redo_steps:
        raise SystemExit("--redo-only requires at least one step (via --redo-only or --redo)")

    catalog = load_video_catalog(catalog_path)
    blob_index = {v.get("blob"): v for v in catalog if isinstance(v, dict) and v.get("blob")}
    selected_paths = []
    if getattr(args, "path", None):
        direct_path = Path(args.path)
        if not direct_path.exists():
            resolved = resolve_video_arg(args.path, blob_index, videos_dir)
            direct_path = resolved or direct_path
        selected_paths.append(direct_path)

    if getattr(args, "video", None):
        selected_paths.extend(select_videos(args, catalog, videos_dir))

    if selected_paths:
        selected_paths = list(dict.fromkeys(selected_paths))
    else:
        selected_paths = select_videos(args, catalog, videos_dir)

    if not selected_paths:
        raise SystemExit("No videos selected. Use: python3 main.py process <path>")
    if args.command == "rag" and len(selected_paths) != 1:
        raise SystemExit("RAG supports exactly one video. Use --video to pick one.")

    test_videos = {make_output_dir(p, processed_root): str(p) for p in selected_paths}
    rag_only = args.command == "rag"
    if redo_only_steps and getattr(args, "redo", None):
        redo_steps = list(dict.fromkeys(redo_steps + _flatten(args.redo)))
    redo_only = redo_only_flag

    embedding_provider, embedding_model = resolve_embedding_config(
        getattr(args, "embedding_provider", None),
        getattr(args, "embedding_model", None),
    )
    debug_enabled = bool(getattr(args, "debug", False)) and not bool(getattr(args, "quiet", False))
    quiet = bool(getattr(args, "quiet", False))

    plan_path = Path("log_reports") / "PARALLELIZATION_PLAN.md"
    benchmark_path = Path("log_reports") / "PARALLELIZATION_BENCHMARKS.md"
    ensure_benchmark_plan(plan_path)

    for output_dir, test_video in test_videos.items():
        run_started = time.perf_counter()
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        checkpoint_path = f"{output_dir}/checkpoint.json"

        storage_manager.__init__(
            chat_id=getattr(args, "chat_id", None),
            mongo_uri=mongo_uri,
            local_path=Path(checkpoint_path),
            video_name=test_video,
        )

        if rag_only:
            rag_path = f"{output_dir}/rag_embedding.json"
            if not os.path.exists(rag_path):
                emit(f"RAG embedding not found: {rag_path}. Run process first.", quiet=quiet, force=True)
                continue
            ask_rag(
                rag_path=rag_path,
                show_k_context=True,
                k=rag_top_k_context,
                conv_path=f"{output_dir}/conversation_history.json",
                log_source=checkpoint_path,
                show_timings=False,
            )
            continue

        run_params = dict(params)
        run_params.update({
            "execution_mode": args.execution_mode,
            "debug": debug_enabled,
            "quiet": quiet,
            "embedding_provider": embedding_provider,
            "embedding_model": embedding_model,
            "low_mem_mode": LOW_MEM_MODE,
        })

        log = initiate_log(
            video_path=test_video,
            run_description="Production pipeline benchmark run.",
            params=run_params,
        )

        checkpoint = storage_manager.read_checkpoint()
        checkpoint.setdefault("steps", {})
        step = checkpoint["steps"]

        if redo_steps:
            checkpoint, redo_info = apply_redo(
                checkpoint=checkpoint,
                output_dir=output_dir,
                redo_steps=redo_steps,
                redo_only=redo_only,
            )
            if redo_info.get("changed") and "scenes" in checkpoint:
                storage_manager.save_checkpoint(checkpoint)

        checkpoint = ensure_scene_detection(
            checkpoint=checkpoint,
            step_store=step,
            storage_manager=storage_manager,
            test_video=test_video,
            output_dir=output_dir,
            debug=debug_enabled,
            quiet=quiet,
        )

        checkpoint = run_visual_audio_pipeline(
            checkpoint=checkpoint,
            step_store=step,
            storage_manager=storage_manager,
            test_video=test_video,
            output_dir=output_dir,
            execution_mode=args.execution_mode,
            debug=debug_enabled,
            quiet=quiet,
        )

        checkpoint = run_llm_and_rag_pipeline(
            checkpoint=checkpoint,
            step_store=step,
            storage_manager=storage_manager,
            test_video=test_video,
            output_dir=output_dir,
            debug=debug_enabled,
            quiet=quiet,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
        )

        total_wall_time = time.perf_counter() - run_started
        checkpoint.setdefault("benchmark", {})
        checkpoint["benchmark"] = {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "execution_mode": args.execution_mode,
            "debug": debug_enabled,
            "quiet": quiet,
            "low_mem_mode": LOW_MEM_MODE,
            "embedding_provider": embedding_provider,
            "embedding_model": embedding_model,
            "total_wall_time_sec": round(total_wall_time, 5),
        }

        complete = complete_log(
            log=log,
            steps=step,
            vid_len=checkpoint.get("video_duration_sec"),
            scene_num=len(checkpoint.get("scenes", [])),
        )
        checkpoint["run_log"] = complete

        save_log(data=checkpoint, path=f"logs/{output_dir}.json")
        storage_manager.save_checkpoint(checkpoint=checkpoint)

        append_benchmark_report(
            benchmark_path,
            {
                "timestamp": checkpoint["benchmark"]["timestamp"],
                "video_name": Path(test_video).name,
                "video_path": test_video,
                "execution_mode": args.execution_mode,
                "debug": debug_enabled,
                "quiet": quiet,
                "low_mem_mode": LOW_MEM_MODE,
                "embedding_provider": embedding_provider,
                "embedding_model": embedding_model,
                "total_wall_time_sec": total_wall_time,
            },
            step,
        )


if __name__ == "__main__":
    main()
