"""End-to-end pipeline test on a real sample video.

Runs all pipeline stages sequentially: scene detection → frame sampling →
BLIP captioning → YOLO detection → audio (AST + ASR) → LLM scene description →
narrative summary → synopsis.

Requires: API keys, GPU/models, and test fixture video.
"""

import os

import pytest

from kairos.config import PipelineConfig

pytestmark = pytest.mark.e2e


@pytest.fixture
def fast_config():
    return PipelineConfig.fast()


@pytest.fixture
def llm_client():
    from kairos.llm.client import build_llm_client
    try:
        client = build_llm_client()
    except Exception as e:
        pytest.skip(f"Could not build LLM client: {e}")
    return client


def test_full_pipeline(sample_video_path, fast_config, llm_client, tmp_path):
    cfg = fast_config
    client = llm_client
    video_path = str(sample_video_path)
    output_dir = str(tmp_path / "output")
    os.makedirs(output_dir, exist_ok=True)

    # --- Stage 1: Scene detection ---
    from kairos.video.scene_detection import get_scene_list

    scenes = get_scene_list(
        input_video_path=video_path,
        threshold=cfg.pyscene_threshold,
        min_scene_sec=cfg.pyscene_shortest,
    )
    assert isinstance(scenes, list)
    assert len(scenes) >= 1, "No scenes detected"
    for s in scenes:
        assert "scene_index" in s
        assert "start_seconds" in s
        assert "end_seconds" in s

    # --- Stage 2: Frame sampling ---
    from kairos.video.frame_sampling import sample_frames

    scenes = sample_frames(
        input_video_path=video_path,
        scenes=scenes,
        num_frames=cfg.frames_per_scene,
        new_size=cfg.frame_resolution,
        output_dir=f"{output_dir}/.frames",
    )
    assert all("frames" in s for s in scenes), "Frames not sampled"

    # --- Stage 3: BLIP captioning ---
    from kairos.video.frame_captioning import caption_frames

    scenes = caption_frames(
        scenes=scenes,
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
        debug=False,
    )
    for s in scenes:
        assert "frame_captions" in s, "Missing frame_captions"
        assert isinstance(s["frame_captions"], list)
        assert all(isinstance(c, str) and len(c) > 0 for c in s["frame_captions"])

    # --- Stage 4: YOLO detection ---
    from kairos.video.frame_sampling import sample_fps
    from kairos.video.object_detection import detect_object_yolo

    scenes = sample_fps(
        input_video_path=video_path,
        scenes=scenes,
        fps=cfg.yolo_action_fps,
        new_size=cfg.frame_resolution,
        output_dir=f"{output_dir}/.fps",
        frames_key="yolo_frames",
        frame_paths_key="yolo_frame_paths",
    )

    scenes = detect_object_yolo(
        scenes=scenes,
        model_size=cfg.yolo_model_path,
        conf=cfg.yolo_conf_thres,
        iou=cfg.yolo_iou_thres,
        output_dir=f"{output_dir}/.yolo",
        frame_key="yolo_frames",
        summary_key="yolo_detections",
        debug=False,
    )
    for s in scenes:
        assert "yolo_detections" in s, "Missing yolo_detections"

    # --- Stage 5: Audio ---
    from kairos.audio.prescan import scan_audio
    from kairos.audio.transcription import extract_speech_singlecall
    from kairos.audio.classifier import extract_sounds_optimized

    scan_result = scan_audio(
        video_path=video_path,
        scenes=scenes,
        target_sr=cfg.asr_target_sr,
        debug=False,
    )

    asr_result = extract_speech_singlecall(
        scenes=scenes,
        scan_result=scan_result,
        model_size=cfg.asr_model_size,
        use_vad=cfg.asr_use_vad,
        debug=False,
    )
    scenes = asr_result[0] if isinstance(asr_result, tuple) else asr_result
    for s in scenes:
        assert "audio_speech" in s, "Missing audio_speech"

    ast_result = extract_sounds_optimized(
        scenes=scenes,
        scan_result=scan_result,
        debug=False,
    )
    scenes = ast_result[0] if isinstance(ast_result, tuple) else ast_result
    for s in scenes:
        assert "audio_natural" in s, "Missing audio_natural"

    # --- Stage 6: LLM scene description ---
    from kairos.llm.scene_description import describe_scenes

    scenes = describe_scenes(
        scenes=scenes,
        client=client,
        hist_size=cfg.llm_scene_history,
        YOLO_key="yolo_detections",
        FLIP_key="frame_captions",
        ASR_key="audio_speech",
        AST_key="audio_natural",
        SUMMARY_key="llm_scene_description",
        cooldown_sec=0,
        debug=False,
        video_path=video_path,
    )
    assert len(scenes) >= 1
    for s in scenes:
        assert "llm_scene_description" in s, "Missing llm_scene_description"
        assert isinstance(s["llm_scene_description"], str)
        assert len(s["llm_scene_description"]) > 0

    # --- Stage 7: Narrative summary ---
    from kairos.llm.synopsis import summarize_scenes

    checkpoint = {"scenes": scenes, "steps": {}}
    checkpoint = summarize_scenes(
        client=client,
        scenes=scenes,
        chunk_size=cfg.llm_chunk_len,
        summary_len=cfg.llm_summary_len,
        debug=False,
        output_dir=output_dir,
    )
    assert "narratives" in checkpoint, "Missing narratives"

    # --- Stage 8: Synopsis ---
    from kairos.llm.synopsis import synthesize_synopsis

    checkpoint = synthesize_synopsis(
        client=client,
        data=checkpoint,
        debug=False,
        output_dir=output_dir,
    )
    assert "synopsis" in checkpoint, "Missing synopsis"
