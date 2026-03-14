"""CLI entry point: parse args, select videos, dispatch to pipeline or RAG."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from kairos.config import PipelineConfig
from kairos.llm.client import build_llm_client
from kairos.cli.args import parse_args
from kairos.cli.catalog import load_video_catalog, select_videos, make_output_dir


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


def main():
    VIDEOS_DIR = Path("data/videos")
    CATALOG_PATH = VIDEOS_DIR / "_all_videos.json"
    PROCESSED_ROOT = Path("data/processed")

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

    client = build_llm_client(llm=getattr(args, "llm", None))

    for output_dir, video_path in test_videos.items():
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        if rag_only:
            _run_rag(output_dir, cfg)
            continue

        from kairos.core.pipeline import run_pipeline
        run_pipeline(
            video_path=video_path,
            output_dir=output_dir,
            cfg=cfg,
            client=client,
            redo_steps=redo_steps or None,
            redo_only=redo_only,
        )


def _run_rag(output_dir: str, cfg: PipelineConfig):
    from kairos.llm.rag import ask_rag

    rag_path = f"{output_dir}/rag_embedding.json"
    checkpoint_path = f"{output_dir}/checkpoint.json"
    if not os.path.exists(rag_path):
        print(f"RAG embedding not found: {rag_path}. Run process first.")
        return
    ask_rag(
        rag_path=rag_path, show_k_context=True, k=cfg.rag_top_k_context,
        conv_path=f"{output_dir}/conversation_history.json",
        log_source=checkpoint_path, show_timings=False,
    )
