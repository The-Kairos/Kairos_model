import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from main import get_llm_client, model_name
from src.debug_utils import save_checkpoint
from src.kg_node_list import merge_scene_relationships
from src.log_utils import (
    kg_extract_spatial_log,
    kg_extract_temporal_log,
    kg_sync_neo4j_log,
)


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


def clear_extended_outputs(checkpoint: dict) -> dict:
    scenes = checkpoint.get("scenes", [])
    if isinstance(scenes, list):
        for scene in scenes:
            if not isinstance(scene, dict):
                continue
            scene.pop("spatial_relationships", None)
            scene.pop("spatial_relationship_errors", None)
            scene.pop("temporal_relationships", None)
            scene.pop("temporal_relationship_errors", None)

    steps = checkpoint.get("steps")
    if isinstance(steps, dict):
        steps.pop("kg_extract_spatial", None)
        steps.pop("kg_extract_temporal", None)
        steps.pop("kg_sync_neo4j", None)

    return checkpoint


def all_scenes_have_key(scenes: list[dict], key: str) -> bool:
    return bool(scenes) and all(isinstance(scene, dict) and key in scene for scene in scenes)


def merge_extended_relationships(scenes: list[dict]) -> list[dict]:
    merged_scenes = []
    for scene in scenes:
        new_scene = dict(scene)
        new_scene["relationships"] = merge_scene_relationships(
            scene.get("relationships", []),
            scene.get("spatial_relationships", []),
            scene.get("temporal_relationships", []),
        )
        merged_scenes.append(new_scene)
    return merged_scenes


def build_variant_summary(variant: str, checkpoint: dict, checkpoint_path: Path, error: str | None = None) -> dict:
    scenes = checkpoint.get("scenes", []) if isinstance(checkpoint, dict) else []
    knowledge_graph = checkpoint.get("knowledge_graph", {}) if isinstance(checkpoint, dict) else {}
    return {
        "variant": variant,
        "checkpoint_path": str(checkpoint_path),
        "status": "failure" if error else "success",
        "error": error,
        "scene_count": len(scenes),
        "all_scenes_have_relationships": all_scenes_have_key(scenes, "relationships"),
        "all_scenes_have_spatial_relationships": all_scenes_have_key(scenes, "spatial_relationships"),
        "all_scenes_have_temporal_relationships": all_scenes_have_key(scenes, "temporal_relationships"),
        "neo4j": knowledge_graph.get("neo4j"),
    }


def run_extended_variant(checkpoint_path: Path, video_path: str, database_name_seed: str) -> dict:
    checkpoint = clear_extended_outputs(load_checkpoint(checkpoint_path))
    checkpoint.setdefault("steps", {})
    scenes = checkpoint.get("scenes", [])
    known_nodes = checkpoint.get("knowledge_graph", {}).get("nodes", {})

    if not scenes:
        raise RuntimeError("Checkpoint has no scenes.")
    if not known_nodes:
        raise RuntimeError("Checkpoint is missing knowledge_graph.nodes.")
    if not all_scenes_have_key(scenes, "relationships"):
        raise RuntimeError("Checkpoint is missing base narrative relationships. Run the original ablation script first.")

    llm_client = get_llm_client()

    checkpoint["scenes"], checkpoint["steps"]["kg_extract_spatial"] = kg_extract_spatial_log(
        scenes=scenes,
        known_nodes=known_nodes,
        client=llm_client,
        model=model_name,
    )
    checkpoint["scenes"], checkpoint["steps"]["kg_extract_temporal"] = kg_extract_temporal_log(
        scenes=checkpoint["scenes"],
        known_nodes=known_nodes,
        client=llm_client,
        model=model_name,
    )
    checkpoint["scenes"] = merge_extended_relationships(checkpoint["scenes"])
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


def run_extended_ablation(
    video_path: str = DEFAULT_VIDEO_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict:
    output_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "video_path": video_path,
        "output_root": str(output_root),
        "variants": [],
    }

    for variant in VARIANTS:
        checkpoint_path = output_root / variant / "checkpoint.json"
        database_name_seed = f"{Path(video_path).name}__{variant}"
        emit(f"[extended] {variant}")
        try:
            checkpoint = run_extended_variant(checkpoint_path, video_path, database_name_seed)
            summary["variants"].append(build_variant_summary(variant, checkpoint, checkpoint_path))
        except Exception as exc:
            checkpoint = load_checkpoint(checkpoint_path) if checkpoint_path.exists() else {}
            summary["variants"].append(
                build_variant_summary(
                    variant,
                    checkpoint,
                    checkpoint_path,
                    error=f"Extended KG stage failed: {type(exc).__name__}: {exc}",
                )
            )
            emit(f"[extended][failed] {variant}: {type(exc).__name__}: {exc}")
        else:
            emit(f"[extended][done] {variant}")

    summary_path = output_root / "ablation_summary_extended.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    emit(f"Summary written to: {summary_path}")
    return summary


if __name__ == "__main__":
    run_extended_ablation()
