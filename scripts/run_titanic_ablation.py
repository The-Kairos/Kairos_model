import json
import sys
from copy import deepcopy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from main import get_llm_client, llm_cooldown_sec, llm_scene_history, model_name
from src.debug_utils import save_checkpoint
from src.log_utils import describe_scenes_log, kg_extract_log, kg_sync_neo4j_log


DEFAULT_SOURCE_CHECKPOINT = PROJECT_ROOT / "_processed" / "Titanic.1997.mkv" / "checkpoint.json"
DEFAULT_VIDEO_PATH = "Titanic.1997.mkv"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "_processed_ablations" / "Titanic.1997.mkv"
VARIANTS = ("full", "no_yolo", "no_asr", "no_ast", "no_blip")


def emit(message: str) -> None:
    print(message, flush=True)


def load_checkpoint(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_checkpoint(checkpoint: dict, path: Path) -> None:
    save_checkpoint(checkpoint=checkpoint, path=path)


def all_scenes_have_key(scenes: list[dict], key: str) -> bool:
    return bool(scenes) and all(isinstance(scene, dict) and key in scene for scene in scenes)


def clear_downstream_outputs(checkpoint: dict) -> dict:
    scenes = checkpoint.get("scenes", [])
    if isinstance(scenes, list):
        for scene in scenes:
            if not isinstance(scene, dict):
                continue
            scene.pop("llm_scene_description", None)
            scene.pop("relationships", None)

    checkpoint.pop("knowledge_graph", None)
    checkpoint.pop("narratives", None)
    checkpoint.pop("synopsis", None)
    checkpoint.pop("rag_embedding", None)
    checkpoint.pop("benchmark", None)
    checkpoint.pop("run_log", None)

    steps = checkpoint.get("steps")
    if isinstance(steps, dict):
        for key in (
            "describe_scenes",
            "kg_extract",
            "kg_sync_neo4j",
            "summarize_scenes",
            "synthesize_synopsis",
            "make_embedding",
        ):
            steps.pop(key, None)

    return checkpoint


def apply_variant_mutation(base_checkpoint: dict, variant: str) -> dict:
    checkpoint = deepcopy(base_checkpoint)
    checkpoint = clear_downstream_outputs(checkpoint)
    scenes = checkpoint.get("scenes", [])

    if variant == "full":
        return checkpoint

    key_by_variant = {
        "no_yolo": "yolo_detections",
        "no_asr": "audio_speech",
        "no_ast": "audio_natural",
        "no_blip": "frame_captions",
    }
    empty_value_by_variant = {
        "no_yolo": [],
        "no_asr": "",
        "no_ast": "",
        "no_blip": [],
    }
    target_key = key_by_variant[variant]
    replacement_value = empty_value_by_variant[variant]

    for scene in scenes:
        if not isinstance(scene, dict):
            continue
        scene[target_key] = replacement_value

    return checkpoint


def ensure_variant_checkpoint(base_checkpoint: dict, output_root: Path, variant: str) -> Path:
    variant_dir = output_root / variant
    variant_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = variant_dir / "checkpoint.json"
    checkpoint = apply_variant_mutation(base_checkpoint, variant)
    write_checkpoint(checkpoint, checkpoint_path)
    return checkpoint_path


def run_llm_stage(checkpoint_path: Path, video_path: str) -> dict:
    checkpoint = load_checkpoint(checkpoint_path)
    checkpoint.setdefault("steps", {})
    scenes = checkpoint.get("scenes", [])
    if not scenes:
        raise RuntimeError("Checkpoint has no scenes.")

    if all_scenes_have_key(scenes, "llm_scene_description") and checkpoint.get("knowledge_graph", {}).get("nodes"):
        return checkpoint

    llm_client = get_llm_client()
    describe_output, step_log = describe_scenes_log(
        scenes=scenes,
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
        return_metadata=True,
        debug=False,
        video_path=video_path,
    )
    checkpoint["scenes"], describe_metadata = describe_output
    checkpoint.setdefault("knowledge_graph", {})
    checkpoint["knowledge_graph"]["nodes"] = (
        describe_metadata.get("knowledge_graph", {}).get("nodes", {})
    )
    checkpoint["steps"]["describe_scenes"] = step_log
    write_checkpoint(checkpoint, checkpoint_path)
    return checkpoint


def run_kg_stage(checkpoint_path: Path, video_path: str, database_name_seed: str) -> dict:
    checkpoint = load_checkpoint(checkpoint_path)
    checkpoint.setdefault("steps", {})
    scenes = checkpoint.get("scenes", [])
    known_nodes = checkpoint.get("knowledge_graph", {}).get("nodes", {})
    if not scenes:
        raise RuntimeError("Checkpoint has no scenes.")
    if not known_nodes:
        raise RuntimeError("Checkpoint is missing knowledge_graph.nodes.")

    if not all_scenes_have_key(scenes, "relationships"):
        checkpoint["scenes"], kg_log = kg_extract_log(
            scenes=scenes,
            known_nodes=known_nodes,
            client=get_llm_client(),
            model=model_name,
        )
        checkpoint["steps"]["kg_extract"] = kg_log
        write_checkpoint(checkpoint, checkpoint_path)

    neo4j_meta, neo4j_log = kg_sync_neo4j_log(
        video_path=video_path,
        scenes=checkpoint["scenes"],
        known_nodes=known_nodes,
        database_name_seed=database_name_seed,
    )
    checkpoint.setdefault("knowledge_graph", {})
    checkpoint["knowledge_graph"]["neo4j"] = neo4j_meta
    checkpoint["steps"]["kg_sync_neo4j"] = neo4j_log
    write_checkpoint(checkpoint, checkpoint_path)
    return checkpoint


def build_variant_summary(variant: str, checkpoint: dict, checkpoint_path: Path, error: str | None = None) -> dict:
    scenes = checkpoint.get("scenes", []) if isinstance(checkpoint, dict) else []
    knowledge_graph = checkpoint.get("knowledge_graph", {}) if isinstance(checkpoint, dict) else {}
    return {
        "variant": variant,
        "checkpoint_path": str(checkpoint_path),
        "status": "failure" if error else "success",
        "error": error,
        "scene_count": len(scenes),
        "all_scenes_have_llm_scene_description": all_scenes_have_key(scenes, "llm_scene_description"),
        "all_scenes_have_relationships": all_scenes_have_key(scenes, "relationships"),
        "has_knowledge_graph_nodes": bool(knowledge_graph.get("nodes")),
        "neo4j": knowledge_graph.get("neo4j"),
    }


def run_ablation(
    source_checkpoint: Path = DEFAULT_SOURCE_CHECKPOINT,
    video_path: str = DEFAULT_VIDEO_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict:
    base_checkpoint = load_checkpoint(source_checkpoint)
    output_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "source_checkpoint": str(source_checkpoint),
        "video_path": video_path,
        "output_root": str(output_root),
        "variants": [],
    }
    checkpoint_paths: dict[str, Path] = {}

    emit("=== Step 1/3: Preparing all variants ===")
    for variant in VARIANTS:
        emit(f"[prepare] {variant}")
        checkpoint_path = ensure_variant_checkpoint(base_checkpoint, output_root, variant)
        checkpoint_paths[variant] = checkpoint_path

    emit("=== Step 2/3: Running LLM scene descriptions for all variants ===")
    for variant in VARIANTS:
        checkpoint_path = checkpoint_paths[variant]
        try:
            emit(f"[llm] {variant}")
            checkpoint = run_llm_stage(checkpoint_path, video_path)
        except Exception as exc:
            checkpoint = load_checkpoint(checkpoint_path)
            summary["variants"].append(build_variant_summary(
                variant,
                checkpoint,
                checkpoint_path,
                error=f"LLM stage failed: {type(exc).__name__}: {exc}",
            ))
            emit(f"[llm][failed] {variant}: {type(exc).__name__}: {exc}")
        else:
            emit(f"[llm][done] {variant}")

    emit("=== Step 3/3: Running KG extraction and Neo4j sync for all variants ===")
    final_variants: list[dict] = []
    existing_errors = {entry["variant"]: entry for entry in summary["variants"]}
    summary["variants"] = []

    for variant in VARIANTS:
        checkpoint_path = checkpoint_paths[variant]
        database_name_seed = f"{Path(video_path).name}__{variant}"
        if variant in existing_errors:
            final_variants.append(existing_errors[variant])
            emit(f"[kg][skipped] {variant} because LLM stage failed earlier")
            continue
        try:
            emit(f"[kg] {variant}")
            checkpoint = run_kg_stage(checkpoint_path, video_path, database_name_seed)
            final_variants.append(build_variant_summary(variant, checkpoint, checkpoint_path))
        except Exception as exc:
            checkpoint = load_checkpoint(checkpoint_path)
            final_variants.append(build_variant_summary(
                variant,
                checkpoint,
                checkpoint_path,
                error=f"KG stage failed: {type(exc).__name__}: {exc}",
            ))
            emit(f"[kg][failed] {variant}: {type(exc).__name__}: {exc}")
        else:
            emit(f"[kg][done] {variant}")

    summary["variants"] = final_variants

    summary_path = output_root / "ablation_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    emit(f"Summary written to: {summary_path}")
    return summary


def main() -> None:
    summary = run_ablation()
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
