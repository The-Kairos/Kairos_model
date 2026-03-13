from __future__ import annotations

import base64
import os
import time
from pathlib import Path
from typing import List, Tuple

from dotenv import load_dotenv

load_dotenv()

from openai import AzureOpenAI
from openai import APIConnectionError, BadRequestError


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FRAME_ROOT = Path(__file__).resolve().parent / "frame_boundaries"
REPORT_PATH = Path(__file__).resolve().parent / "_test_results.md"
REPORT_MARKER = "## GPT-4o Reports"


def _client() -> AzureOpenAI:
    endpoint = os.getenv("GPT_ENDPOINT")
    deployment = os.getenv("GPT_DEPLOYMENT")
    subscription_key = os.getenv("GPT_KEY")
    api_version = os.getenv("GPT_VERSION")

    if not endpoint or not deployment or not subscription_key or not api_version:
        missing = [
            name
            for name, value in [
                ("GPT_ENDPOINT", endpoint),
                ("GPT_DEPLOYMENT", deployment),
                ("GPT_KEY", subscription_key),
                ("GPT_VERSION", api_version),
            ]
            if not value
        ]
        raise RuntimeError(f"Missing env vars: {', '.join(missing)}")

    client = AzureOpenAI(
        api_version=api_version,
        azure_endpoint=endpoint,
        api_key=subscription_key,
    )
    return client


def _data_url(image_path: Path) -> str:
    data = image_path.read_bytes()
    encoded = base64.b64encode(data).decode("utf-8")
    suffix = image_path.suffix.lower().lstrip(".")
    if suffix not in {"jpg", "jpeg", "png"}:
        suffix = "jpeg"
    return f"data:image/{suffix};base64,{encoded}"


def _list_video_dirs() -> List[Path]:
    if not FRAME_ROOT.exists():
        return []
    return sorted([p for p in FRAME_ROOT.iterdir() if p.is_dir()])


def _list_concat_images(video_dir: Path) -> List[Path]:
    images = [
        p
        for p in video_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    return sorted(images, key=lambda p: p.name.lower())


def _pipeline_basics() -> str:
    return (
        "Kairos pipeline basics (from main.py):\n"
        "- PySceneDetect splits video into scenes (threshold + min scene length).\n"
        "- Frames are sampled per scene and resized.\n"
        "- BLIP captions sampled frames.\n"
        "- YOLOv8 detects objects on sampled FPS frames.\n"
        "- MIT AST extracts non-speech audio events.\n"
        "- Whisper ASR extracts speech.\n"
        "- GPT-4o creates scene descriptions, then summary narrative and synopsis.\n"
        "- RAG embeddings are built from the processed scenes.\n"
        "Goal: scene splits should be semantically coherent, not too many, not too few, "
        "and should support BLIP/YOLO/AST/ASR and downstream GPT summaries."
    )


def _build_messages(video_name: str, images: List[Path], results_md: str) -> List[dict]:
    system_prompt = (
        "You are a video segmentation expert helping tune a scene-splitting step "
        "for a multi-stage vision/audio NLP pipeline. Avoid any sexual content; "
        "if sensitive content appears, keep descriptions high-level and safe. "
        "Prefer methods that capture many distinct scenes when those scenes "
        "represent different events or visually different moments. Penalize "
        "methods that combine multiple distinct events into one scene."
    )
    user_intro = (
        f"You will be given contact sheets for segmentation methods for the video: {video_name}.\n"
        f"{_pipeline_basics()}\n\n"
        "Existing test results (for context, do not repeat verbatim):\n"
        f"{results_md}\n\n"
        "Task:\n"
        "1) Judge quality by whether each scene is visually distinct and represents a different event.\n"
        "2) It is GOOD if the method captures many scenes *when those scenes are truly different events*.\n"
        "3) Under-segmentation is BAD: combining different events/visuals into one scene is harmful.\n"
        "4) Over-segmentation is BAD: splitting within a continuous shot/event is harmful.\n"
        "5) Recommend the best method for the Kairos pipeline and briefly justify.\n"
        "6) Call out any method that clearly over-segments or under-segments and why.\n"
        "7) If a close second exists, mention it and why.\n"
        "Return a concise report (max ~200 words)."
    )

    content: List[dict] = [{"type": "text", "text": user_intro}]
    for img in images:
        content.append({"type": "text", "text": f"Method: {img.stem}"})
        content.append(
            {"type": "image_url", "image_url": {"url": _data_url(img)}}
        )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]


def _load_results_md() -> str:
    if REPORT_PATH.exists():
        return REPORT_PATH.read_text(encoding="utf-8")
    return ""


def _strip_existing_reports(content: str) -> str:
    if REPORT_MARKER not in content:
        return content.rstrip()
    return content.split(REPORT_MARKER, 1)[0].rstrip()


def _build_report_section(reports: List[Tuple[str, str]]) -> str:
    lines = [REPORT_MARKER, ""]
    for video_name, report in reports:
        lines.append(f"### {video_name}")
        lines.append("")
        lines.append(report.strip())
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _write_reports(base_md: str, reports: List[Tuple[str, str]]) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cleaned = _strip_existing_reports(base_md)
    report_block = _build_report_section(reports)
    combined = f"{cleaned}\n\n{report_block}".lstrip()
    REPORT_PATH.write_text(combined, encoding="utf-8")


def _format_filter_error(err: BadRequestError) -> str:
    try:
        details = err.response.json()
        inner = details.get("error", {}).get("inner_error", {})
        return inner.get("code") or "content_filter"
    except Exception:
        return "content_filter"


def _safe_report(reason: str) -> str:
    return f"cannot process cuz of {reason}"


def main() -> None:
    client = _client()
    deployment = os.getenv("GPT_DEPLOYMENT")
    video_dirs = _list_video_dirs()
    if not video_dirs:
        raise SystemExit(f"No video folders found in {FRAME_ROOT}")

    base_md = _load_results_md()
    reports: List[Tuple[str, str]] = []

    for video_dir in video_dirs:
        images = _list_concat_images(video_dir)
        if not images:
            print(f"Skipping {video_dir.name}: no concatenated images found.")
            continue

        messages = _build_messages(video_dir.name, images, base_md)
        try:
            response = client.chat.completions.create(
                messages=messages,
                max_tokens=800,
                temperature=0.2,
                top_p=1.0,
                model=deployment,
            )
            report = response.choices[0].message.content or ""
            reports.append((video_dir.name, report))
            print(f"Wrote report for {video_dir.name}")
            continue
        except BadRequestError as err:
            reason = _format_filter_error(err)
            print(f"{video_dir.name}: filtered ({reason}), retrying once with safer prompt.")
            safer_messages = _build_messages(
                video_dir.name,
                images,
                "Do not include any explicit content. Keep discussion abstract and safe.",
            )
            try:
                response = client.chat.completions.create(
                    messages=safer_messages,
                    max_tokens=800,
                    temperature=0.2,
                    top_p=1.0,
                    model=deployment,
                )
                report = response.choices[0].message.content or ""
                reports.append((video_dir.name, report))
                print(f"Wrote report for {video_dir.name}")
                continue
            except BadRequestError as err2:
                reason2 = _format_filter_error(err2)
                reports.append((video_dir.name, _safe_report(reason2)))
                print(f"{video_dir.name}: filtered again ({reason2}).")
                continue
        except APIConnectionError:
            print(f"{video_dir.name}: connection error, retrying in 5s.")
            time.sleep(5)
            try:
                response = client.chat.completions.create(
                    messages=messages,
                    max_tokens=800,
                    temperature=0.2,
                    top_p=1.0,
                    model=deployment,
                )
                report = response.choices[0].message.content or ""
                reports.append((video_dir.name, report))
                print(f"Wrote report for {video_dir.name}")
            except Exception as err:
                reports.append((video_dir.name, _safe_report(type(err).__name__)))
                print(f"{video_dir.name}: failed after retry ({type(err).__name__}).")
            continue

    _write_reports(base_md, reports)


if __name__ == "__main__":
    main()
