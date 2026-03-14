"""Generate markdown summary reports from pipeline log JSON files."""

from __future__ import annotations

import argparse
import glob
import json
import os

import pandas as pd

DEFAULT_LOG_DIR = "./logs/runs"
DEFAULT_OUTPUT_DIR = "./logs/reports"
DEFAULT_OUTPUT_MD = "new_report.md"

STEP_KEYS = {
    "get_scene_list": ("video_length", "PySceneDetect*"),
    "ast_timings": ("video_length", "AST sound descriptions*"),
    "asr_timings": ("video_length", "ASR speech transcription*"),
    "save_clips": ("scene_number", "Masked clips saving"),
    "sample_frames": ("scene_number", "Frame sampling"),
    "caption_frames": ("scene_number", "BLIP caption"),
    "detect_object_yolo": ("video_length", "YOLO detection*"),
    "describe_scenes": (
        "scene_number",
        "BLIP + YOLO + AST + ASR in GPT4o",
    ),
    "summarize_scenes": ("video_length", "Summarization*"),
    "synthesize_synopsis": ("video_length", "Synopsis + common Q&A*"),
}

METRIC_COLUMNS = [
    "wall_time_sec",
    "wall_time_%",
    "cpu_time_sec",
    "ram_used_MB",
    "io_read_MB",
    "io_write_MB",
]


def to_number(value: object) -> float | str | object:
    """Convert *value* to a float, parsing ``HH:MM:SS`` if needed."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        if ":" in value:
            parts = value.split(":")
            if len(parts) == 3:
                try:
                    h, m, s = (float(p) for p in parts)
                    return h * 3600 + m * 60 + s
                except ValueError:
                    return value
        try:
            return float(value)
        except ValueError:
            return value
    return value


def safe_div(x: float | int, d: float | int | None) -> float | int:
    """Divide *x* by *d*, returning *x* unchanged when *d* is 0/None."""
    return x / d if d not in (0, None) else x


def format_num(value: object, precision: int = 2, fallback: str = "n/a") -> str:
    """Format a number to *precision* decimal places."""
    if isinstance(value, (int, float)):
        return f"{value:.{precision}f}"
    return fallback


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate markdown summary from log JSON files."
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        default=DEFAULT_LOG_DIR,
        help="Folder containing log JSON files.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Output folder for markdown (and optional CSVs).",
    )
    parser.add_argument(
        "-m",
        "--output-md",
        default=DEFAULT_OUTPUT_MD,
        help="Output markdown filename.",
    )
    parser.add_argument(
        "--save-csv",
        action="store_true",
        default=False,
        help="Also write per-video CSVs.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for the log-report CLI."""
    args = _parse_args()

    log_dir = args.input_dir
    output_dir = args.output_dir
    output_md = os.path.join(output_dir, args.output_md)
    save_csv = args.save_csv

    os.makedirs(output_dir, exist_ok=True)

    markdown_sections: list[str] = []
    json_files = glob.glob(os.path.join(log_dir, "*.json"))

    for file_path in json_files:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            document = json.load(f)

        video_path = document.get("video_path", "unknown")
        video_title = os.path.basename(video_path)
        scene_count = to_number(document.get("scene_number", 1))
        llm_cooldown_sec = document.get("params", {}).get("llm_cooldown_sec", 5)
        total_sec = to_number(document.get("total_process_sec", 1))

        rows: list[dict] = []
        for step_key, (divisor_key, friendly_name) in STEP_KEYS.items():
            step_data = document.get("steps", {}).get(step_key, {})
            row: dict = {"step": friendly_name}

            divisor_value = to_number(document.get(divisor_key, 1))
            raw_wall_time = to_number(step_data.get("wall_time_sec", 0))

            for metric in METRIC_COLUMNS:
                if metric == "wall_time_%":
                    if (
                        isinstance(raw_wall_time, (int, float))
                        and isinstance(total_sec, (int, float))
                        and total_sec > 0
                    ):
                        row[metric] = safe_div(raw_wall_time, total_sec) * 100
                    else:
                        row[metric] = raw_wall_time
                    continue

                raw_value = to_number(step_data.get(metric, 0))

                if isinstance(raw_value, (int, float)) and isinstance(
                    divisor_value, (int, float)
                ):
                    row[metric] = safe_div(raw_value, divisor_value)
                else:
                    row[metric] = raw_value

                if (
                    step_key == "describe_scenes"
                    and metric == "wall_time_sec"
                    and isinstance(row[metric], (int, float))
                ):
                    row[metric] -= llm_cooldown_sec

                if (
                    step_key
                    in (
                        "get_scene_list",
                        "ast_timings",
                        "asr_timings",
                    )
                    and metric != "wall_time_%"
                ):
                    if isinstance(row[metric], (int, float)):
                        row[metric] *= 60

            rows.append(row)

        df = pd.DataFrame(rows)
        max_wall_pct = None
        if "wall_time_%" in df.columns:
            numeric_wall = [v for v in df["wall_time_%"] if isinstance(v, (int, float))]
            if numeric_wall:
                max_wall_pct = max(numeric_wall)

        for col in METRIC_COLUMNS:
            if col == "wall_time_%":

                def fmt_wall_pct(
                    x: object,
                    _max: float | None = max_wall_pct,
                ) -> object:
                    if isinstance(x, (int, float)):
                        formatted = f"{x:.1f}%"
                        if _max is not None and x == _max:
                            return f"**{formatted}**"
                        return formatted
                    return x

                df[col] = df[col].apply(fmt_wall_pct)
            else:
                df[col] = df[col].apply(
                    lambda x: f"{x:.3f}" if isinstance(x, (int, float)) else x
                )

        if "wall_time_%" in df.columns:
            ordered_cols = ["wall_time_%", "step"] + [
                c for c in df.columns if c not in ("wall_time_%", "step")
            ]
            df = df[ordered_cols]

        base_name = os.path.splitext(video_title)[0].replace(" ", "_")
        csv_path = os.path.join(output_dir, f"{base_name}.csv")

        if save_csv:
            df.to_csv(csv_path, index=False)

        synopsis = document.get("synopsis")

        md = f"## {video_title}\n\n"
        if synopsis:
            summary_text = None
            if isinstance(synopsis, dict):
                summary_text = synopsis.get("summary")
            elif isinstance(synopsis, str):
                parts = [p.strip() for p in synopsis.split("\n\n") if p.strip()]
                if parts:
                    summary_text = parts[0]
            if isinstance(summary_text, str) and summary_text.strip():
                md += f"{summary_text.strip()}\n\n"

        colalign = ["center", "left"] + ["right"] * (len(df.columns) - 2)
        md += df.to_markdown(index=False, colalign=colalign)
        md += "\n\n"

        video_length = to_number(document.get("video_length", 1))

        if isinstance(scene_count, (int, float)) and isinstance(
            total_sec, (int, float)
        ):
            run_without_delay = total_sec - (llm_cooldown_sec * scene_count)
        else:
            run_without_delay = total_sec

        if (
            isinstance(video_length, (int, float))
            and video_length > 0
            and isinstance(run_without_delay, (int, float))
        ):
            k = run_without_delay / video_length
        else:
            k = 0

        md += (
            f"**Footnote:**  \n"
            f"`total_process_sec` without LLM cooldown"
            f" ({format_num(llm_cooldown_sec)}s per scene,"
            f" {format_num(run_without_delay)}s total)"
            f" is **{format_num(k)}x longer** than"
            f" `video_length` of {format_num(video_length)}s.\n"
            f"**{scene_count} scenes** were detected"
            f" in `{video_path}`\n"
            f"\\* measured per minute of video, whereas the"
            f" remaining processes are measured per scenes.\n"
        )

        markdown_sections.append(md)

    with open(output_md, "w", encoding="utf-8") as f:
        f.write("# Processing Logs Summary\n\n")
        for section in markdown_sections:
            f.write(section)

    if save_csv:
        print(f"Done! CSVs + Markdown generated in {output_md}")
    else:
        print(f"Done! Markdown generated in {output_md}")


if __name__ == "__main__":
    main()
