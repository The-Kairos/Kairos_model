from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Iterable, List

from PIL import Image, ImageDraw, ImageFont

BASE_DIR = Path(__file__).resolve().parent
FRAME_ROOT = BASE_DIR / "frame_boundaries"
RESULTS_PATH = BASE_DIR / "_test_logs.json"

_SCENE_RE = re.compile(r"scene(\d+)", re.IGNORECASE)


def _scene_index(path: Path) -> int:
    match = _SCENE_RE.search(path.stem)
    if match:
        return int(match.group(1))
    digits = "".join(ch for ch in path.stem if ch.isdigit())
    return int(digits) if digits else 0


def _sanitize_filename(name: str) -> str:
    invalid = '<>:"/\\|?*'
    cleaned = "".join("_" if ch in invalid else ch for ch in name)
    return cleaned.strip().strip(".")


def _collect_scene_images(test_dir: Path) -> List[Path]:
    images = [p for p in test_dir.glob("scene*.jpg") if p.is_file()]
    if not images:
        images = [p for p in test_dir.glob("*.jpg") if p.is_file()]
    return sorted(images, key=_scene_index)


def _resize_image(image: Image.Image, scale: float) -> Image.Image:
    if scale == 1.0:
        return image
    new_w = max(1, int(round(image.width * scale)))
    new_h = max(1, int(round(image.height * scale)))
    return image.resize((new_w, new_h), Image.LANCZOS)


def _load_results(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _iter_result_sets(results: dict) -> List[dict]:
    if isinstance(results, dict) and isinstance(results.get("videos"), list):
        return [r for r in results["videos"] if isinstance(r, dict)]
    if isinstance(results, dict):
        return [results]
    return []


def _index_results_by_video(results: dict) -> dict:
    indexed: dict[str, dict] = {}
    for entry in _iter_result_sets(results):
        video = entry.get("video")
        if not isinstance(video, str):
            continue
        key = _sanitize_filename(Path(video).stem)
        indexed[key] = entry
    return indexed


def _lookup_metrics(results: dict, test_name: str, fallback_count: int) -> tuple[int, float]:
    if not isinstance(results, dict):
        return fallback_count, 0.0
    entry = results.get(test_name)
    if not isinstance(entry, dict):
        return fallback_count, 0.0
    scenes = entry.get("scenes", [])
    count = len(scenes) if isinstance(scenes, list) else fallback_count
    elapsed = entry.get("elapsed_seconds", 0.0)
    try:
        elapsed_val = float(elapsed)
    except (TypeError, ValueError):
        elapsed_val = 0.0
    return count, elapsed_val


def _lookup_threshold(results: dict, test_name: str) -> str | None:
    if not isinstance(results, dict):
        return None
    entry = results.get(test_name)
    if not isinstance(entry, dict):
        return None
    for key in ("threshold", "score_threshold", "merge_threshold", "pyscene_threshold"):
        if key in entry:
            try:
                value = float(entry[key])
            except (TypeError, ValueError):
                return None
            return f"{value:.2f}"
    return None


def _lookup_timecode(results: dict, test_name: str, scene_idx: int) -> str | None:
    if not isinstance(results, dict):
        return None
    entry = results.get(test_name)
    if not isinstance(entry, dict):
        return None
    scenes = entry.get("scenes", [])
    if not isinstance(scenes, list) or not scenes:
        return None
    if 0 <= scene_idx < len(scenes):
        scene = scenes[scene_idx]
        if isinstance(scene, dict):
            start_seconds = scene.get("start_seconds")
            if start_seconds is not None:
                try:
                    total = float(start_seconds)
                    mins = int(total // 60)
                    secs = total - mins * 60
                    return f"{mins:02d}:{secs:04.1f}"
                except (TypeError, ValueError):
                    return None
            value = scene.get("start_timecode")
            if value:
                return str(value)
    return None


def _text_height(font: ImageFont.ImageFont) -> int:
    bbox = font.getbbox("Ag")
    return bbox[3] - bbox[1]


def build_contact_sheet(
    test_dir: Path,
    video_name: str,
    results: dict,
    columns: int = 5,
    scale: float = 0.5,
    gap: int = 8,
    row_gap: int | None = None,
) -> Path | None:
    images = _collect_scene_images(test_dir)
    if not images:
        return None

    first = Image.open(images[0]).convert("RGB")
    tile = _resize_image(first, scale)
    tile_w, tile_h = tile.size

    rows = int(math.ceil(len(images) / float(columns)))
    if row_gap is None:
        row_gap = gap * 2
    font = ImageFont.load_default()
    scene_count, elapsed = _lookup_metrics(results, test_dir.name, len(images))
    threshold = _lookup_threshold(results, test_dir.name)
    header_lines = [
        f"Video: {video_name}",
        (
            f"Test: {test_dir.name} | Scenes: {scene_count} | "
            f"Time: {elapsed:.2f}s | Threshold: {threshold if threshold is not None else 'n/a'}"
        ),
    ]
    header_height = _text_height(font) * len(header_lines) + gap

    sheet_w = tile_w * columns + gap * (columns + 1)
    sheet_h = header_height + tile_h * rows + row_gap * (rows + 1)
    sheet = Image.new("RGB", (sheet_w, sheet_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(sheet)

    y = gap
    for line in header_lines:
        draw.text((gap, y), line, fill=(0, 0, 0), font=font)
        y += _text_height(font)

    for idx, path in enumerate(images):
        with Image.open(path) as img:
            img = img.convert("RGBA")
            img = _resize_image(img, scale)
            timecode = _lookup_timecode(results, test_dir.name, _scene_index(path))
            if timecode:
                text_h = _text_height(font)
                overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
                draw_tile = ImageDraw.Draw(overlay)
                text_w = draw_tile.textlength(timecode, font=font)
                pad = 2
                draw_tile.rectangle(
                    (0, 0, text_w + pad * 2, text_h + pad * 2),
                    fill=(0, 0, 0, 128),
                )
                draw_tile.text((pad, pad), timecode, fill=(255, 255, 255, 255), font=font)
                img = Image.alpha_composite(img, overlay)
            img = img.convert("RGB")
        row = idx // columns
        col = idx % columns
        x = gap + col * (tile_w + gap)
        y = header_height + row_gap + row * (tile_h + row_gap)
        sheet.paste(img, (x, y))

    output_path = test_dir.parent / f"{test_dir.name}.jpg"
    sheet.save(output_path, quality=92)
    return output_path


def build_contact_sheets(
    frame_root: Path = FRAME_ROOT,
    columns: int = 5,
    scale: float = 0.5,
    gap: int = 8,
    results_path: Path | None = None,
) -> List[Path]:
    outputs: List[Path] = []
    if not frame_root.exists():
        return outputs

    results_data = _load_results(results_path or RESULTS_PATH)
    results_by_video = _index_results_by_video(results_data)

    for video_dir in sorted(p for p in frame_root.iterdir() if p.is_dir()):
        video_results = results_by_video.get(video_dir.name, {})
        for test_dir in sorted(p for p in video_dir.iterdir() if p.is_dir()):
            output = build_contact_sheet(
                test_dir,
                video_name=video_dir.name,
                results=video_results,
                columns=columns,
                scale=scale,
                gap=gap,
            )
            if output is not None:
                outputs.append(output)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build contact sheets for scene boundary frames."
    )
    parser.add_argument(
        "--root",
        default=str(FRAME_ROOT),
        help="Root folder containing video/test subfolders.",
    )
    parser.add_argument(
        "--columns",
        type=int,
        default=5,
        help="Number of columns in the contact sheet.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0.5,
        help="Scale factor for each tile (default 0.5).",
    )
    parser.add_argument(
        "--gap",
        type=int,
        default=8,
        help="Gap between tiles in pixels (default 8).",
    )
    parser.add_argument(
        "--results",
        default=str(RESULTS_PATH),
        help="Path to _test_logs.json for scene counts and timings.",
    )
    args = parser.parse_args()

    outputs = build_contact_sheets(
        frame_root=Path(args.root),
        columns=max(1, args.columns),
        scale=max(0.1, args.scale),
        gap=max(0, args.gap),
        results_path=Path(args.results),
    )
    if outputs:
        print("Built contact sheets:")
        for path in outputs:
            print(path)
    else:
        print("No contact sheets created (no images found).")


if __name__ == "__main__":
    main()
