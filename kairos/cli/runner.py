"""Pipeline orchestration loop."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from kairos.config import PipelineConfig
from kairos.core.checkpoint import read_json, save_checkpoint, have_key, save_clips
from kairos.core.logging import log_step, initiate_log, complete_log, save_log
from kairos.core.redo import apply_redo
from kairos.core.utils import print_section, see_scenes_cuts
from kairos.llm.client import build_llm_client
from kairos.cli.args import parse_args
from kairos.cli.catalog import load_video_catalog, select_videos, make_output_dir


def _logged(func):
    """Apply log_step decorator inline."""
    return log_step()(func)


def main():
    VIDEOS_DIR = Path("Videos")
    CATALOG_PATH = VIDEOS_DIR / "_all_videos.json"
    PROCESSED_ROOT = Path("_processed")

    args = parse_args()

    preset = getattr(args, "preset", "default")
    cfg = {
        "fast": PipelineConfig.fast,
        "motion": PipelineConfig.motion_sensitive,
        "static": PipelineConfig.static_video,
        "default": PipelineConfig.default,
    }.get(preset, PipelineConfig.default)()

    redo_only_flag = getattr(args, "redo_only", None) is not None
    redo_only_steps = getattr(args, "redo_only", None) or []
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

    client, model_name, deployment = build_llm_client(llm=getattr(args, "llm", None))

    # Lazy imports — models only loaded when their step runs
    from kairos.video.scene_detection import get_scene_list
    from kairos.video.frame_sampling import sample_frames, sample_fps
    from kairos.video.frame_captioning import caption_frames
    from kairos.video.object_detection import detect_object_yolo
    from kairos.llm.scene_description import describe_scenes
    from kairos.audio.detector import scan_audio
    from kairos.audio.classifier import extract_sounds_optimized
    from kairos.audio.transcription import extract_speech_singlecall
    from kairos.llm.synopsis import summarize_scenes, synthesize_synopsis
    from kairos.llm.rag import make_embedding, ask_rag

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
            asr_result, step["asr_timings"] = _logged(extract_speech_singlecall)(
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
            checkpoint["scenes"] = asr_result[0] if isinstance(asr_result, tuple) else asr_result
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
