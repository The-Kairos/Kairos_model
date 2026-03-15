# Kairos CLI Reference

Kairos exposes four CLI entry points installed via `pyproject.toml`:

| Command | Entry point | Purpose |
|---------|-------------|---------|
| `kairos process` | `kairos.cli.app:main` | Run the full video-processing pipeline |
| `kairos rag` | `kairos.cli.app:main` | Interactive RAG chatbot for a processed video |
| `kairos-download` | `kairos.cli.download:main` | Download test videos from the catalog |
| `kairos-report` | `kairos.cli.report:main` | Generate markdown reports from pipeline logs |
| `kairos-compare` | `kairos.cli.compare:main` | Compare LLM scene descriptions across log files |

---

## `kairos process`

Run the video-processing pipeline on one or more videos. The pipeline executes:
scene detection → frame sampling → BLIP captioning → YOLO detection →
audio analysis (Whisper + AST) → LLM scene descriptions → narrative → synopsis → RAG embedding.

### Flags

| Flag | Type | Description |
|------|------|-------------|
| `--video <name>` | repeatable | Blob name or file path of a video to process. Can be specified multiple times. |
| `--all` | flag | Process every video in the catalog (`_all_videos.json`). |
| `--filter <level>` | choice | Filter videos by duration: `short` (<10 min), `medium` (<30 min), `long` (<90 min), `extra` (all). |
| `--include-unknown` | flag | Include videos whose length is unknown when using `--filter`. |
| `--preset <name>` | choice | Pipeline configuration preset: `default`, `fast`, `motion`, `static`. |
| `--llm <backend>` | choice | LLM backend: `gemini`, `openai`, or `claude`. Overrides the `LLM_BACKEND` env var. |
| `--redo <step> [step …]` | repeatable | Re-run a pipeline step **and all downstream dependents** (transitive redo). |
| `--redo-only [step …]` | optional list | Re-run **only** the specified steps without propagating to dependents (non-transitive). |

### Presets

| Preset | Description |
|--------|-------------|
| `default` | Balanced settings for general-purpose videos. |
| `fast` | Higher scene threshold (40), 1 frame/scene, large LLM chunks, 8 parallel workers. |
| `motion` | Lower scene threshold (15), 5 frames/scene, 8 FPS YOLO sampling — for action-heavy content. |
| `static` | Very low scene threshold (3), 1 frame/scene, 0.5 FPS YOLO — for slow or static content. |

### Redo Steps

Available step names for `--redo` and `--redo-only`:

```
scenes, frame_captions, yolo, audio_natural, audio_speech, llm, narrative, synopsis, rag
```

**Dependency chain:**

```
scenes         → frame_captions, yolo, audio_natural, audio_speech, llm, narrative, synopsis, rag
frame_captions → llm → narrative → synopsis → rag
yolo           → llm → narrative → synopsis → rag
audio_natural  → llm → narrative → synopsis → rag
audio_speech   → llm → narrative → synopsis → rag
```

When using `--redo`, all downstream dependents are cleared and re-run.
When using `--redo-only`, only the exact steps listed are re-run.

### Examples

```bash
# Process a single video with default settings
kairos process --video "Young Sheldon - First Day of High School.mp4"

# Process all videos in the catalog
kairos process --all

# Process only short videos (< 10 min)
kairos process --filter short

# Process medium videos including those with unknown length
kairos process --filter medium --include-unknown

# Process with the fast preset and Gemini as the LLM backend
kairos process --video "Titanic.1997.mkv" --preset fast --llm gemini

# Redo the LLM descriptions and all downstream steps (narrative, synopsis, rag)
kairos process --video "CCTV Dogs.mp4" --redo llm

# Redo only frame captions without re-running downstream steps
kairos process --video "CCTV Dogs.mp4" --redo-only frame_captions

# Redo multiple steps transitively
kairos process --video "CCTV Dogs.mp4" --redo yolo --redo audio_speech

# Process multiple videos at once
kairos process --video "Video1.mp4" --video "Video2.mp4"
```

---

## `kairos rag`

Start an interactive RAG (Retrieval-Augmented Generation) chatbot for a **single** previously-processed video. The video must have been processed through the full pipeline first to generate the `rag_embedding.json` file.

### Flags

| Flag | Type | Description |
|------|------|-------------|
| `--video <name>` | required | Blob name or file path of the video to query. |
| `--llm <backend>` | choice | LLM backend: `gemini`, `openai`, or `claude`. Overrides `LLM_BACKEND` env var. |

### Interactive Session

Once started, the chatbot prompts for questions and returns answers grounded in the video's scene descriptions, synopsis, and metadata. Type `exit` or `quit` to end the session.

Conversation history is automatically saved to `conversation_history.json` in the video's output directory.

### Examples

```bash
# Start RAG session for a processed video
kairos rag --video "Young Sheldon - First Day of High School.mp4"

# Use Claude as the generation backend
kairos rag --video "Titanic.1997.mkv" --llm claude

# Use Gemini for generation
kairos rag --video "CCTV Dogs.mp4" --llm gemini
```

**Example interaction:**

```
RAG ready. Ask questions (type 'exit' to quit).

Question: What happens in the opening scene?
================================================================================
Answer:
The video opens with a shot of a young boy standing outside a large school
building (00:00:01 - 00:00:15). He appears nervous as other students walk
past him toward the entrance...
================================================================================

Question: exit
```

---

## `kairos-download`

Interactive CLI tool for downloading test videos from the JSON catalog (`data/videos/_all_videos.json`). Videos are stored in `data/videos/`.

### Workflow

1. **Category selection** — choose which videos to download by duration:
   - `1` — Short: under 10 minutes
   - `2` — Medium: up to 30 minutes
   - `3` — Long: up to 90 minutes
   - `4` — Extra: all video lengths
   - `5` — Cheatsheet only (generates CLI cheatsheet without downloading)

2. **Unknown-length handling** — if any videos have unknown length, you are prompted whether to include them.

3. **Download + probing** — each video is downloaded (skipped if already on disk), then probed with `ffprobe` (fallback: OpenCV, moviepy) for duration and resolution.

4. **Catalog & log updates** — `_all_videos.json` and `_logs.json` are updated with duration, resolution, download times, and SAS link expiry.

5. **Cheatsheet generation** — a markdown cheatsheet (`.cli_cheatsheet.md`) is generated with ready-to-run commands for processing and RAG.

### Examples

```bash
# Launch the interactive downloader
kairos-download

# Example session:
# ====== Choose which videos to download: ======
# 1) Short: under 10 minutes
# 2) Medium: up to 30 minutes
# 3) Long: up to 90 minutes
# 4) Extra: all video lengths
# 5) Cheatsheet only (no downloads)
# ==============================================
# Option (1-5): 1
# >> Downloading: Young Sheldon - First Day of High School.mp4
# >> Downloading: CCTV Dogs.mp4
# =================== Summary ==================
# Downloaded : 2
# Skipped    : 0
# Folder     : /path/to/data/videos
# ==============================================
# Cheatsheet : data/.cli_cheatsheet.md
```

---

## `kairos-report`

Generate a consolidated markdown performance report from pipeline JSON log files. The report includes per-step timing, CPU usage, RAM, and I/O metrics for each processed video.

### Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `-i`, `--input-dir` | path | `./logs/runs` | Folder containing log JSON files. |
| `-o`, `--output-dir` | path | `./logs/reports` | Output folder for the markdown report (and optional CSVs). |
| `-m`, `--output-md` | filename | `new_report.md` | Output markdown filename. |
| `--save-csv` | flag | `false` | Also write per-video CSV files alongside the report. |

### Metrics Reported

For each video, the report includes a table with these columns per pipeline step:

- `wall_time_%` — Percentage of total processing time
- `wall_time_sec` — Wall-clock time (normalized per minute or per scene)
- `cpu_time_sec` — CPU time
- `ram_used_MB` — RAM usage
- `io_read_MB` / `io_write_MB` — Disk I/O

Steps marked with `*` are measured per minute of video; others are measured per scene.

### Examples

```bash
# Generate a report from default log directory
kairos-report

# Custom input/output directories
kairos-report -i ./logs/runs -o ./logs/reports -m my_report.md

# Generate report with per-video CSV exports
kairos-report --save-csv

# Full example with all options
kairos-report -i ./logs/runs -o ./analysis -m benchmark_report.md --save-csv
```

---

## `kairos-compare`

Compare LLM scene descriptions across multiple log files and export the comparison to an Excel workbook. Log files are grouped by filename prefix (e.g., all logs starting with the same video name), and each group gets its own Excel sheet.

### Behavior

1. Reads all `*.json` files from `./logs/runs`.
2. Groups files by the first `_`-delimited token of their filename.
3. For each group, creates an Excel sheet with:
   - One column per log file containing the `llm_scene_description` for each scene.
   - Shared columns: `frame_captions`, `yolo_detections`, `audio_natural`, `audio_speech`.
4. Output is written to `./logs/reports/llm_descriptions_comparisons.xlsx`.

### Examples

```bash
# Generate the comparison Excel file
kairos-compare

# Output: ./logs/reports/llm_descriptions_comparisons.xlsx
```

---

## Environment Variables

All CLI commands respect these environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_BACKEND` | `openai` | Default LLM backend (`gemini`, `openai`, `claude`) |
| `OPENAI_KEY` | — | OpenAI / Azure OpenAI API key |
| `OPENAI_ENDPOINT` | — | OpenAI API base URL |
| `OPENAI_MODEL` | `gpt-4o` | OpenAI model or deployment name |
| `GEMINI_PROJECT` | `prj-udst-prod-oussama-1` | GCP project for Vertex AI |
| `GEMINI_LOCATION` | `us-central1` | GCP region for Vertex AI |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Gemini model for generation |
| `GEMINI_EMBEDDING_MODEL` | `gemini-embedding-001` | Gemini model for RAG embeddings |
| `GEMINI_RAG_MODEL` | `gemini-2.5-pro` | Gemini model for RAG answer generation |
| `CLAUDE_LOCATION` | `us-east5` | GCP region for Claude via Vertex AI |
| `CLAUDE_PROJECT` | (falls back to `GEMINI_PROJECT`) | GCP project for Claude |
| `CLAUDE_MODEL` | `claude-sonnet-4-6` | Claude model identifier |
| `WHISPER_API_KEY` | — | Azure OpenAI key for Whisper API |
| `WHISPER_API_ENDPOINT` | — | Azure OpenAI endpoint for Whisper |
| `WHISPER_API_VERSION` | `2024-12-01-preview` | Azure API version for Whisper |
| `WHISPER_API_DEPLOYMENT` | — | Azure deployment name for Whisper |
