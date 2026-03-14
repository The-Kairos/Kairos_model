"""Pipeline step orchestration: scene detection -> frame sampling -> captioning -> YOLO -> audio -> LLM -> synopsis -> RAG."""

from __future__ import annotations

import os
from pathlib import Path

from kairos.config import PipelineConfig
from kairos.core.checkpoint import have_key, read_json, save_checkpoint, save_clips
from kairos.core.logging import complete_log, initiate_log, log_step, save_log
from kairos.core.redo import apply_redo
from kairos.core.utils import print_prefixed, print_section, see_scenes_cuts


def _logged(func):
    """Apply log_step decorator inline."""
    return log_step()(func)


# ---------------------------------------------------------------------------
# Step-group helpers — each owns its logging and checkpoint saves
# ---------------------------------------------------------------------------


def _run_scene_detection(checkpoint, step, video_path, cfg, checkpoint_path):
    from kairos.video.scene_detection import get_scene_list

    if checkpoint.get("scenes"):
        return

    print_section("Running PysceneDetect...")
    checkpoint["scenes"], step["get_scene_list"] = _logged(get_scene_list)(
        input_video_path=video_path,
        threshold=cfg.pyscene_threshold,
        min_scene_sec=cfg.pyscene_shortest,
    )
    see_scenes_cuts(df=checkpoint["scenes"])

    clips_dir = f"{os.path.dirname(checkpoint_path)}/.clips"
    print_prefixed("(Pipeline)", f"Saving clips in: {clips_dir}")
    checkpoint["scenes"], step["save_clips"] = _logged(save_clips)(
        video_path=video_path,
        scenes=checkpoint["scenes"],
        output_dir=clips_dir,
    )
    save_checkpoint(checkpoint=checkpoint, path=checkpoint_path)


def _run_frame_processing(checkpoint, step, video_path, cfg, checkpoint_path):
    from kairos.video.frame_captioning import caption_frames
    from kairos.video.frame_sampling import sample_frames

    if have_key(checkpoint["scenes"], "frame_captions"):
        return

    output_dir = os.path.dirname(checkpoint_path)
    frames_dir = f"{output_dir}/.frames"
    print_prefixed("(Pipeline)", f"Saving sampled frames in: {frames_dir}")
    checkpoint["scenes"], step["sample_frames"] = _logged(sample_frames)(
        input_video_path=video_path,
        scenes=checkpoint["scenes"],
        num_frames=cfg.frames_per_scene,
        new_size=cfg.frame_resolution,
        output_dir=frames_dir,
    )

    print_section("Running BLIP...")
    checkpoint["scenes"], step["caption_frames"] = _logged(caption_frames)(
        scenes=checkpoint["scenes"],
        debug=True,
        **cfg.blip_params,
    )
    save_checkpoint(checkpoint=checkpoint, path=checkpoint_path)


def _run_yolo(checkpoint, step, video_path, cfg, checkpoint_path):
    from kairos.video.frame_sampling import sample_fps
    from kairos.video.object_detection import detect_object_yolo

    if have_key(checkpoint["scenes"], "yolo_detections"):
        return

    output_dir = os.path.dirname(checkpoint_path)

    if not have_key(checkpoint["scenes"], "yolo_frames"):
        fps_dir = f"{output_dir}/.fps"
        print_prefixed("(Pipeline)", f"Saving sampled fps in: {fps_dir}")
        checkpoint["scenes"], step["sample_fps"] = _logged(sample_fps)(
            input_video_path=video_path,
            scenes=checkpoint["scenes"],
            fps=cfg.yolo_action_fps,
            new_size=cfg.frame_resolution,
            output_dir=fps_dir,
            frames_key="yolo_frames",
            frame_paths_key="yolo_frame_paths",
        )

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


def _run_audio(checkpoint, step, video_path, cfg, checkpoint_path):
    from kairos.audio.classifier import extract_sounds_optimized
    from kairos.audio.prescan import scan_audio
    from kairos.audio.transcription import extract_speech_singlecall

    scan_result = None

    def _ensure_scan():
        nonlocal scan_result
        if scan_result is not None:
            return scan_result
        print_section("Running Audio Pre-Scan...")
        scan_result, step["audio_scan"] = _logged(scan_audio)(
            video_path=video_path,
            scenes=checkpoint["scenes"],
            target_sr=cfg.asr_target_sr,
            debug=True,
        )
        return scan_result

    if not have_key(checkpoint["scenes"], "audio_speech"):
        scan_result = _ensure_scan()
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
        checkpoint["scenes"] = (
            asr_result[0] if isinstance(asr_result, tuple) else asr_result
        )
        save_checkpoint(checkpoint=checkpoint, path=checkpoint_path)

    if not have_key(checkpoint["scenes"], "audio_natural"):
        scan_result = _ensure_scan()
        print_section("Running MIT AST (Parallel)...")
        scenes_result, step["ast_timings"] = _logged(extract_sounds_optimized)(
            scenes=checkpoint["scenes"],
            scan_result=scan_result,
            max_workers=4,
            use_processes=True,
            force_cpu=False,
            debug=True,
        )
        checkpoint["scenes"] = (
            scenes_result[0] if isinstance(scenes_result, tuple) else scenes_result
        )
        save_checkpoint(checkpoint=checkpoint, path=checkpoint_path)


def _run_llm(checkpoint, step, video_path, cfg, client, checkpoint_path, output_dir):
    from kairos.llm.scene_description import describe_scenes
    from kairos.llm.synopsis import summarize_scenes, synthesize_synopsis

    if not have_key(checkpoint["scenes"], "llm_scene_description"):
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
            cooldown_sec=cfg.llm_cooldown_sec,
            debug=True,
            video_path=video_path,
        )
        save_checkpoint(checkpoint=checkpoint, path=checkpoint_path)

    if "narratives" not in checkpoint:
        print_section("Running LLM Summary narrative...")
        checkpoint, step["summarize_scenes"] = _logged(summarize_scenes)(
            client=client,
            scenes=checkpoint["scenes"],
            chunk_size=cfg.llm_chunk_len,
            summary_len=cfg.llm_summary_len,
            debug=True,
            output_dir=output_dir,
        )
        narratives = checkpoint.get("narratives", [])
        if narratives:
            last = narratives[-1]
            narrative_path = (
                Path(output_dir)
                / f"narrative_{len(narratives)}_len_{last['narrative_len']}.txt"
            )
            print_prefixed("(Pipeline)", f"Saving narrative in: {narrative_path}")
        save_checkpoint(checkpoint=checkpoint, path=checkpoint_path)

    if "synopsis" not in checkpoint:
        print_section("Running LLM Synopsis generation...")
        checkpoint, step["synthesize_synopsis"] = _logged(synthesize_synopsis)(
            client=client,
            data=checkpoint,
            debug=True,
            output_dir=output_dir,
        )
        save_checkpoint(checkpoint=checkpoint, path=checkpoint_path)


def _run_rag(checkpoint, step, output_dir, checkpoint_path, log, video_path):
    from kairos.llm.rag import make_embedding

    rag_path = f"{output_dir}/rag_embedding.json"
    if os.path.exists(rag_path):
        return

    checkpoint["rag_embedding"], step["make_embedding"] = _logged(make_embedding)(
        checkpoint=checkpoint,
        output_path=rag_path,
    )

    cleared_checkpoint = save_checkpoint(checkpoint=checkpoint, path=checkpoint_path)
    log = complete_log(
        log=log,
        steps=step,
        vid_len=checkpoint["scenes"][-1]["end_timecode"],
        scene_num=len(checkpoint["scenes"]),
        vid_df=cleared_checkpoint,
    )

    save_log(data=log, path=f"logs/runs/{output_dir}.json")
    save_checkpoint(checkpoint=log, path=checkpoint_path)


# ---------------------------------------------------------------------------
# Main pipeline orchestrator
# ---------------------------------------------------------------------------


def run_pipeline(
    video_path: str,
    output_dir: str,
    cfg: PipelineConfig,
    client,
    redo_steps: list[str] | None = None,
    redo_only: bool = False,
):
    """Run the full video processing pipeline for a single video."""
    redo_steps = redo_steps or []

    log = initiate_log(
        video_path=video_path,
        run_description="Test run for video processing pipeline.",
        params=cfg.to_dict(),
    )

    checkpoint_path = f"{output_dir}/checkpoint.json"
    checkpoint = read_json(json_path=checkpoint_path)
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
            save_checkpoint(checkpoint=checkpoint, path=checkpoint_path)

    _run_scene_detection(checkpoint, step, video_path, cfg, checkpoint_path)
    _run_frame_processing(checkpoint, step, video_path, cfg, checkpoint_path)
    _run_yolo(checkpoint, step, video_path, cfg, checkpoint_path)
    _run_audio(checkpoint, step, video_path, cfg, checkpoint_path)
    _run_llm(checkpoint, step, video_path, cfg, client, checkpoint_path, output_dir)
    _run_rag(checkpoint, step, output_dir, checkpoint_path, log, video_path)
