"""Command-line interface for the Kairos video processing pipeline."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from kairos.config import PipelineConfig
from kairos.checkpoint import read_json, save_checkpoint, have_key, save_clips
from kairos.pipeline_logging import log_step, initiate_log, complete_log, save_log
from kairos.redo import apply_redo, REDO_CHOICES
from kairos.utils import print_section, see_scenes_cuts


# Video catalog helpers

def load_video_catalog(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict) and "videos" in data:
        data = data["videos"]
    if not isinstance(data, list):
        raise ValueError("Expected _all_videos.json to be a list of video objects.")
    return data


def get_video_length_seconds(entry: dict) -> float | None:
    value = entry.get("video_length")
    if isinstance(value, (int, float)) and value > 0:
        return float(value)
    return None


def categorize_length(seconds: float) -> str:
    minutes = seconds / 60
    if minutes < 10:
        return "short"
    if minutes < 30:
        return "medium"
    if minutes < 90:
        return "long"
    return "extra"


def make_output_dir(video_path: Path, processed_root: Path | str = "processed") -> str:
    name = video_path.name
    if name.startswith("."):
        name = name.lstrip(".")
    name = name.strip().rstrip(".")
    if not name:
        name = "video"
    return str(Path(processed_root) / name)


def resolve_video_arg(arg: str, blob_index: dict, videos_dir: Path) -> Path | None:
    candidate = Path(arg)
    if candidate.exists():
        return candidate
    candidate = videos_dir / arg
    if candidate.exists():
        return candidate
    entry = blob_index.get(arg)
    if entry and entry.get("blob"):
        candidate = videos_dir / entry["blob"]
        if candidate.exists():
            return candidate
    return None


def select_videos(args, catalog: list[dict], videos_dir: Path) -> list[Path]:
    blob_index = {v.get("blob"): v for v in catalog if isinstance(v, dict) and v.get("blob")}
    selected_paths: list[Path] = []

    if args.video:
        items = args.video if isinstance(args.video, list) else [args.video]
        for item in items:
            path = resolve_video_arg(item, blob_index, videos_dir)
            if path is None:
                print(f"Skip: video not found: {item}")
                continue
            selected_paths.append(path)
        return selected_paths

    filter_value = getattr(args, "filter", None)
    include_unknown = getattr(args, "include_unknown", False)
    include_all = getattr(args, "all", False)

    if not (include_all or filter_value):
        print("Select videos with --video, --all, or --filter.")
        raise SystemExit(2)

    entries = catalog
    if filter_value:
        rank = {"short": 1, "medium": 2, "long": 3, "extra": 4}
        selected_entries = []
        unknown = 0
        for entry in entries:
            length = get_video_length_seconds(entry)
            if length is None:
                if include_unknown:
                    selected_entries.append(entry)
                else:
                    unknown += 1
                continue
            if rank[categorize_length(length)] <= rank[filter_value]:
                selected_entries.append(entry)
        if unknown and not include_unknown:
            print(f"Skipping {unknown} video(s) with unknown length. Use --include-unknown to include.")
        entries = selected_entries

    for entry in entries:
        blob = entry.get("blob")
        if not blob:
            continue
        path = videos_dir / blob
        if not path.exists():
            print(f"Skip: missing file on disk: {blob}")
            continue
        selected_paths.append(path)
    return selected_paths


# LLM client factory

def _build_llm_client(cfg: PipelineConfig):
    """Build the LLM client and model name/deployment from environment."""
    use_gemini = os.getenv("USE_GEMINI", "").lower() in ("1", "true", "yes")

    if use_gemini:
        from google import genai
        api_key = os.getenv("GEMINI_API_KEY")
        client = genai.Client(vertexai=True, api_key=api_key)
        model_name = "gemini-2.5-flash"
        deployment = None
    else:
        from openai import AzureOpenAI
        endpoint = os.getenv("GPT_ENDPOINT")
        deployment = os.getenv("GPT_DEPLOYMENT")
        subscription_key = os.getenv("GPT_KEY")
        api_version = os.getenv("GPT_VERSION")
        client = AzureOpenAI(api_version=api_version, azure_endpoint=endpoint, api_key=subscription_key)
        model_name = "gpt-4o"

    return client, model_name, deployment


# Argument parsing

def parse_args():
    parser = argparse.ArgumentParser(description="Process videos or run RAG.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    process = subparsers.add_parser("process", help="Process videos")
    process.add_argument("--video", action="append", help="Blob name or path (repeatable)")
    process.add_argument("--all", action="store_true", help="Process all catalog videos")
    process.add_argument("--filter", choices=["short", "medium", "long", "extra"], help="Inclusive length filter")
    process.add_argument("--include-unknown", action="store_true", help="Include videos with unknown length when filtering")
    process.add_argument("--redo", nargs="+", action="append", choices=REDO_CHOICES, help="Redo a processing step (repeatable)")
    process.add_argument("--redo-only", nargs="*", choices=REDO_CHOICES, help="Redo only specified steps (no dependents)")

    rag = subparsers.add_parser("rag", help="Run RAG for a single video")
    rag.add_argument("--video", required=True, help="Blob name or path")

    return parser.parse_args()


# Logged step wrappers (inline decorator application, no boilerplate functions)

def _logged(func):
    """Apply log_step decorator inline."""
    return log_step()(func)


# Main entry point

def main():
    VIDEOS_DIR = Path("Videos")
    CATALOG_PATH = VIDEOS_DIR / "_all_videos.json"
    PROCESSED_ROOT = Path("_processed")

    args = parse_args()
    cfg = PipelineConfig.default()

    redo_only_flag = args.redo_only is not None
    redo_only_steps = args.redo_only or []
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

    catalog = load_video_catalog(CATALOG_PATH)
    selected_paths = select_videos(args, catalog, VIDEOS_DIR)

    if not selected_paths:
        raise SystemExit("No videos selected.")
    if args.command == "rag" and len(selected_paths) != 1:
        raise SystemExit("RAG supports exactly one video. Use --video to pick one.")

    test_videos = {make_output_dir(p, PROCESSED_ROOT): str(p) for p in selected_paths}
    rag_only = args.command == "rag"

    if redo_only_steps and getattr(args, "redo", None):
        redo_steps = list(dict.fromkeys(redo_steps + _flatten(args.redo)))
    redo_only = redo_only_flag

    client, model_name, deployment = _build_llm_client(cfg)

    # Lazy imports — models only loaded when their step runs
    from kairos.scene_detection import get_scene_list
    from kairos.frame_sampling import sample_frames, sample_fps
    from kairos.frame_captioning import caption_frames
    from kairos.object_detection import detect_object_yolo
    from kairos.scene_description import describe_scenes
    from kairos.audio_detector import scan_audio
    from kairos.audio_ast import extract_sounds_optimized
    from kairos.speech_transcription import extract_speech_singlecall
    from kairos.synopsis import summarize_scenes, synthesize_synopsis
    from kairos.rag import make_embedding, ask_rag

    for output_dir, test_video in test_videos.items():
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        if rag_only:
            rag_path = f"{output_dir}/rag_embedding.json"
            checkpoint_path = f"{output_dir}/checkpoint.json"
            if not os.path.exists(rag_path):
                print(f"RAG embedding not found: {rag_path}. Run process first.")
                continue
            ask_rag(
                rag_path=rag_path, show_k_context=True, k=cfg.rag_top_k_context,
                conv_path=f"{output_dir}/conversation_history.json",
                log_source=checkpoint_path, show_timings=False,
            )
            continue

        log = initiate_log(
            video_path=test_video,
            run_description="Test run for video processing pipeline.",
            params=cfg.to_dict(),
        )

        checkpoint_path = f"{output_dir}/checkpoint.json"
        checkpoint = read_json(json_path=checkpoint_path)
        checkpoint.setdefault("steps", {})
        step = checkpoint["steps"]

        if redo_steps:
            checkpoint, redo_info = apply_redo(
                checkpoint=checkpoint, output_dir=output_dir,
                redo_steps=redo_steps, redo_only=redo_only,
            )
            if redo_info.get("changed") and "scenes" in checkpoint:
                save_checkpoint(checkpoint=checkpoint, path=checkpoint_path)

        if not checkpoint.get("scenes"):
            print("")
            print_section("Running PysceneDetect...")
            checkpoint["scenes"], step["get_scene_list"] = _logged(get_scene_list)(
                input_video_path=test_video,
                threshold=cfg.pyscene_threshold,
                min_scene_sec=cfg.pyscene_shortest,
            )
            see_scenes_cuts(df=checkpoint["scenes"])

            print("")
            print(f"Saving clips in: {output_dir}/.clips")
            checkpoint["scenes"], step["save_clips"] = _logged(save_clips)(
                video_path=test_video,
                scenes=checkpoint["scenes"],
                output_dir=f"{output_dir}/.clips",
            )
            save_checkpoint(checkpoint=checkpoint, path=checkpoint_path)

        if not have_key(checkpoint["scenes"], "frame_captions"):
            print(f"Saving sampled frames in: {output_dir}/.frames")
            checkpoint["scenes"], step["sample_frames"] = _logged(sample_frames)(
                input_video_path=test_video,
                scenes=checkpoint["scenes"],
                num_frames=cfg.frames_per_scene,
                new_size=cfg.frame_resolution,
                output_dir=f"{output_dir}/.frames",
            )

            print("")
            print_section("Running BLIP...")
            checkpoint["scenes"], step["caption_frames"] = _logged(caption_frames)(
                scenes=checkpoint["scenes"],
                prompt=cfg.blip_start_prompt,
                max_length=cfg.blip_caption_len,
                min_length=cfg.blip_min_length,
                num_beams=cfg.blip_num_beams,
                do_sample=cfg.blip_do_sample,
                top_p=cfg.blip_top_p,
                temperature=cfg.blip_temperature,
                length_penalty=cfg.blip_length_penalty,
                no_repeat_ngram_size=cfg.blip_no_repeat_ngram_size,
                repetition_penalty=cfg.blip_repetition_penalty,
                debug=True,
            )
            save_checkpoint(checkpoint=checkpoint, path=checkpoint_path)

        if not have_key(checkpoint["scenes"], "yolo_detections"):
            if not have_key(checkpoint["scenes"], "yolo_frames"):
                print("")
                print(f"Saving sampled fps in: {output_dir}/.fps")
                checkpoint["scenes"], step["sample_fps"] = _logged(sample_fps)(
                    input_video_path=test_video,
                    scenes=checkpoint["scenes"],
                    fps=cfg.yolo_action_fps,
                    new_size=cfg.frame_resolution,
                    output_dir=f"{output_dir}/.fps",
                    frames_key="yolo_frames",
                    frame_paths_key="yolo_frame_paths",
                )

            print("")
            print_section("Running YOLOv8...")
            checkpoint["scenes"], step["detect_object_yolo"] = _logged(detect_object_yolo)(
                scenes=checkpoint["scenes"],
                model_size=cfg.yolo_model_path,
                conf=cfg.yolo_conf_thres,
                iou=cfg.yolo_iou_thres,
                output_dir=f"{output_dir}/.yolo",
                frame_key="yolo_frames",
                summary_key="yolo_detections",
                debug=True,
            )
            save_checkpoint(checkpoint=checkpoint, path=checkpoint_path)

        scan_result = None

        def get_scan_result():
            nonlocal scan_result
            print("")
            print_section("Running Audio Pre-Scan...")
            scan_result, step["audio_scan"] = _logged(scan_audio)(
                video_path=test_video,
                scenes=checkpoint["scenes"],
                target_sr=cfg.asr_target_sr,
                debug=True,
            )
            return scan_result

        if not have_key(checkpoint["scenes"], "audio_speech"):
            if scan_result is None:
                scan_result = get_scan_result()
            print("")
            print_section("Running Whisper (Parallel)...")
            checkpoint["scenes"], step["asr_timings"] = _logged(extract_speech_singlecall)(
                scenes=checkpoint["scenes"],
                scan_result=scan_result,
                model_size=cfg.asr_model_size,
                use_vad=cfg.asr_use_vad,
                language=None,
                parallel=True,
                use_api=True,
                force_cpu=False,
                debug=True,
            )
            save_checkpoint(checkpoint=checkpoint, path=checkpoint_path)

        if not have_key(checkpoint["scenes"], "audio_natural"):
            if scan_result is None:
                scan_result = get_scan_result()
            print("")
            print_section("Running MIT AST (Parallel)...")
            scenes_result, step["ast_timings"] = _logged(extract_sounds_optimized)(
                scenes=checkpoint["scenes"],
                scan_result=scan_result,
                max_workers=4,
                use_processes=True,
                force_cpu=False,
                debug=True,
            )
            checkpoint["scenes"] = scenes_result[0] if isinstance(scenes_result, tuple) else scenes_result
            save_checkpoint(checkpoint=checkpoint, path=checkpoint_path)

        if not have_key(checkpoint["scenes"], "llm_scene_description"):
            print("")
            print_section("Running LLM Scene Descriptions...")
            checkpoint["scenes"], step["describe_scenes"] = _logged(describe_scenes)(
                scenes=checkpoint["scenes"],
                client=client,
                hist_size=cfg.llm_scene_history,
                YOLO_key="yolo_detections",
                FLIP_key="frame_captions",
                ASR_key="audio_speech",
                AST_key="audio_natural",
                SUMMARY_key="llm_scene_description",
                model=model_name,
                prompt_path="kairos/prompts/describe_scene.txt",
                cooldown_sec=cfg.llm_cooldown_sec,
                debug=True,
                video_path=test_video,
            )
            save_checkpoint(checkpoint=checkpoint, path=checkpoint_path)

        if "narratives" not in checkpoint:
            print("")
            print_section("Running LLM Summary narrative...")
            checkpoint, step["summarize_scenes"] = _logged(summarize_scenes)(
                client=client,
                deployment=deployment,
                scenes=checkpoint["scenes"],
                chunk_size=cfg.llm_chunk_len,
                summary_len=cfg.llm_summary_len,
                debug=True,
                output_dir=output_dir,
            )
            narratives = checkpoint.get("narratives", [])
            if narratives:
                last = narratives[-1]
                narrative_path = Path(output_dir) / f"narrative_{len(narratives)}_len_{last['narrative_len']}.txt"
                print(f"Saving narrative in: {narrative_path}")
            save_checkpoint(checkpoint=checkpoint, path=checkpoint_path)

        if "synopsis" not in checkpoint:
            print("")
            print_section("Running LLM Synopsis generation...")
            checkpoint, step["synthesize_synopsis"] = _logged(synthesize_synopsis)(
                client=client,
                deployment=deployment,
                data=checkpoint,
                debug=True,
                output_dir=output_dir,
            )
            save_checkpoint(checkpoint=checkpoint, path=checkpoint_path)

        rag_path = f"{output_dir}/rag_embedding.json"
        if not os.path.exists(rag_path):
            checkpoint["rag_embedding"], step["make_embedding"] = _logged(make_embedding)(
                checkpoint=checkpoint,
                output_path=rag_path,
            )

            cleared_checkpoint = save_checkpoint(checkpoint=checkpoint, path=checkpoint_path)
            log = complete_log(
                log=log, steps=step,
                vid_len=checkpoint["scenes"][-1]["end_timecode"],
                scene_num=len(checkpoint["scenes"]),
                vid_df=cleared_checkpoint,
            )

            save_log(data=log, path=f"logs/{output_dir}.json")
            save_checkpoint(checkpoint=log, path=checkpoint_path)


if __name__ == "__main__":
    main()
