import json
import pandas as pd
import glob
import os

LOG_DIR = "./logs"
OUTPUT_DIR = "./log_reports"
OUTPUT_MD = os.path.join(OUTPUT_DIR, f"new_report.md")
SAVE_CSV = False

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

STEP_KEYS = {
    "get_scene_list": ("video_length", "PySceneDetect*"),
    "ast_timings": ("video_length", "AST sound descriptions*"),
    "asr_timings": ("video_length", "ASR speech transcription*"),
    "save_clips": ("scene_number", "Masked clips saving"),
    "sample_frames": ("scene_number", "Frame sampling"),
    "caption_frames": ("scene_number", "BLIP caption"),
    "detect_object_yolo": ("video_length", "YOLO detection*"),
    "describe_scenes": ("scene_number", "BLIP + YOLO + AST + ASR in GPT4o"),
    "summarize_scenes": ("video_length", "Summarization*"),
    "synthesize_synopsis": ("video_length", "Synopsis + common Q&A*"),
}

METRIC_COLUMNS = [
    "wall_time_sec",
    "wall_time_%",
    "cpu_time_sec",
    "ram_used_MB",
    "io_read_MB",
    "io_write_MB"
]


def to_number(value):
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        if ":" in value:
            parts = value.split(":")
            if len(parts) == 3:
                try:
                    hours = float(parts[0])
                    minutes = float(parts[1])
                    seconds = float(parts[2])
                    return hours * 3600 + minutes * 60 + seconds
                except ValueError:
                    return value
        try:
            return float(value)
        except ValueError:
            return value
    return value

def safe_div(x, d):
    return x / d if d not in [0, None] else x

def format_num(value, precision=2, fallback="n/a"):
    if isinstance(value, (int, float)):
        return f"{value:.{precision}f}"
    return fallback

markdown_sections = []

json_files = glob.glob(os.path.join(LOG_DIR, "*.json"))

for file_path in json_files:
    # Fix for UnicodeDecodeError:
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        document = json.load(f)

    video_path = document.get("video_path", "unknown")
    video_title = os.path.basename(video_path)
    scene_count = to_number(document.get("scene_number", 1))
    llm_cooldown_sec = document.get("params", {}).get("llm_cooldown_sec", 5)
    total_sec = to_number(document.get("total_process_sec", 1))

    rows = []

    for step_key, (divisor_key, friendly_name) in STEP_KEYS.items():
        step_data = document.get("steps", {}).get(step_key, {})
        row = {"step": friendly_name}  # Use friendly name here

        divisor_value = to_number(document.get(divisor_key, 1))
        raw_wall_time = to_number(step_data.get("wall_time_sec", 0))

        for metric in METRIC_COLUMNS:
            if metric == "wall_time_%":
                if isinstance(raw_wall_time, (int, float)) and isinstance(total_sec, (int, float)) and total_sec > 0:
                    row[metric] = safe_div(raw_wall_time, total_sec) * 100
                else:
                    row[metric] = raw_wall_time
                continue

            raw_value = to_number(step_data.get(metric, 0))

            if isinstance(raw_value, (int, float)) and isinstance(divisor_value, (int, float)):
                row[metric] = safe_div(raw_value, divisor_value)
            else:
                row[metric] = raw_value

            # Special rule for describe_scenes because of cooldown 
            if step_key == "describe_scenes" and metric == "wall_time_sec":
                if isinstance(row[metric], (int, float)):
                    row[metric] -= llm_cooldown_sec # time.sleep(cooldown) for API

            if step_key in ["get_scene_list", "ast_timings", "asr_timings"] and metric != "wall_time_%":
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
            def fmt_wall_pct(x):
                if isinstance(x, (int, float)):
                    formatted = f"{x:.1f}%"
                    return f"**{formatted}**" if max_wall_pct is not None and x == max_wall_pct else formatted
                return x
            df[col] = df[col].apply(fmt_wall_pct)
        else:
            df[col] = df[col].apply(lambda x: f"{x:.3f}" if isinstance(x, (int, float)) else x)

    # CSV name
    base_name = os.path.splitext(video_title)[0].replace(" ", "_")
    csv_path = os.path.join(OUTPUT_DIR, f"{base_name}.csv")

    if SAVE_CSV:
        df.to_csv(csv_path, index=False)

    synopsis = document.get("synopsis")

    # Markdown section
    md = f"## {video_title}\n\n"
    if synopsis:
        # Use only the first paragraph (summary) from the synopsis text.
        parts = [p.strip() for p in synopsis.split("\n\n") if p.strip()]
        if parts:
            md += f"{parts[0]}\n\n"
    colalign = ["left"] + ["right"] * (len(df.columns) - 1)
    md += df.to_markdown(index=False, colalign=colalign)
    md += "\n\n"

    video_length = to_number(document.get("video_length", 1))

    if isinstance(scene_count, (int, float)) and isinstance(total_sec, (int, float)):
        run_without_delay = total_sec - (llm_cooldown_sec * scene_count)
    else:
        run_without_delay = total_sec

    if isinstance(video_length, (int, float)) and video_length > 0 and isinstance(run_without_delay, (int, float)):
        k = run_without_delay / video_length
    else:
        k = 0

    md += (
        f"**Footnote:**  \n"
        f"`total_process_sec` without LLM cooldown ({format_num(llm_cooldown_sec)}s per scene, {format_num(run_without_delay)}s total) is **{format_num(k)}x longer** than `video_length` of {format_num(video_length)}s.\n"
        f"**{scene_count} scenes** were detected in `{video_path}`\n"
        f"\\* measured per minute of video, whereas the remaining processes are measured per scenes.\n"
    )

    markdown_sections.append(md)

# Write all markdown tables
with open(OUTPUT_MD, "w", encoding="utf-8") as f:
    f.write("# Processing Logs Summary\n\n")
    for section in markdown_sections:
        f.write(section)

if SAVE_CSV:
    print(f"Done! CSVs + Markdown generated in {OUTPUT_MD}")
else:
    print(f"Done! Markdown generated in {OUTPUT_MD}")
