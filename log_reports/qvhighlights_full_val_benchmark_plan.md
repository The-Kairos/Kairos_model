# Full QVHighlights Val Set Benchmark — Batch Processing Plan

## Context

We have publishable results on 51 of 1,519 val-split videos (3.3%). The QVHighlights raw video tarball at `https://nlp.cs.unc.edu/data/jielei/qvh/qvhilights_videos.tar.gz` is **134 GB** and contains ALL videos (train+val+test). We need to download it, extract only val videos, process through Kairos pipeline in batches, and report the full val-set benchmark.

We want only **Moment Retrieval** metrics (not Highlight Detection) since Kairos's highlights are LLM-generated key moments (query-independent), which is a different concept than QVHighlights' query-dependent saliency scores.

**Disk available: 853 GB** — enough for tarball (134 GB) + val videos (~42 GB) + outputs (~2.7 GB).

---

## Quick Answers to User Questions

### What do R@1 and R@5 mean?

- **R@1 (Recall@1):** "Did Kairos's single BEST guess overlap enough with the correct answer?"
  - For each query, look at ONLY the top-1 predicted clip
  - If its IoU with ground truth ≥ threshold (e.g., 0.5) → HIT
  - R@1 = percentage of queries that are HITs

- **R@5 (Recall@5):** "Is the correct answer anywhere in Kairos's top-5 guesses?"
  - For each query, look at the top-5 predicted clips
  - If ANY of them has IoU ≥ threshold → HIT
  - R@5 is more lenient — it's a safety net metric

**Important:** The official QVHighlights evaluation only reports **R1** (not R5). R5 is our internal extra metric. For publishable comparison, we report R1 + mAP.

### Which metrics does the paper report?

**Moment Retrieval only** (what we want):

| Metric | What it measures |
|--------|-----------------|
| R1@0.5 | % of queries where top-1 clip has ≥50% IoU overlap |
| R1@0.7 | % of queries where top-1 clip has ≥70% IoU overlap |
| mAP@0.5 | Mean Average Precision at IoU≥0.5 (considers full ranking) |
| mAP@0.75 | Mean Average Precision at IoU≥0.75 (strict) |
| mAP Avg | Mean of mAP across 10 IoU thresholds (0.5 to 0.95) |

### Which table in the paper to compare against?

**Table 2** in the QVHighlights paper (Lei et al., NeurIPS 2021, arxiv:2107.09609) — titled "Moment Retrieval and Highlight Detection Results on QVHighlights test split."

Note: The paper reports test split results (submitted to eval server). We evaluate on **val split** which is self-evaluated. Published baselines (Moment-DETR, QD-DETR, UniVTG) report both val and test results. We compare against their **val** numbers:

| Method | R1@0.5 | R1@0.7 | mAP Avg |
|--------|--------|--------|---------|
| Moment-DETR (supervised, val) | 52.89 | 33.02 | 25.49 |
| QD-DETR (supervised, val) | 62.40 | 44.98 | 41.22 |
| UniVTG (supervised, val) | 58.86 | 40.86 | 35.47 |

### QVHighlights dataset breakdown

| Split | Queries | Unique Videos | GT available? | Our use |
|-------|---------|---------------|---------------|---------|
| Train | 7,217 | ~7,000 | Yes | Not used (zero-shot) |
| Val | 1,550 | 1,519 | Yes | **This is what we evaluate on** |
| Test | 1,541 | ~1,500 | No (eval server only) | Not used |
| **Total** | **10,308** | **~10,000** | | |

---

## Batch Processing Strategy

### Step 1: Download the tarball

```bash
cd /home/Kairos_model/test/benchmarks/cache
wget -c "https://nlp.cs.unc.edu/data/jielei/qvh/qvhilights_videos.tar.gz" \
     -O qvhilights_videos.tar.gz
```

- Size: **134 GB**
- `-c` enables resume if interrupted
- Estimated time: 30 min to 3 hours depending on bandwidth
- Disk needed: 134 GB (temporary)

### Step 2: Extract ONLY val-split videos

We know exactly which 1,519 video filenames we need (from `highlight_val_release.jsonl`). Write a Python script that reads the tar.gz and extracts ONLY matching files:

```python
import tarfile, json

# Load val video IDs
val_vids = set()
with open("cache/highlight_val_release.jsonl") as f:
    for line in f:
        val_vids.add(json.loads(line)["vid"])

# Extract only val videos from tarball
with tarfile.open("cache/qvhilights_videos.tar.gz", "r:gz") as tar:
    for member in tar:
        name = member.name.split("/")[-1].replace(".mp4", "")
        if name in val_vids:
            tar.extract(member, path="cache/qvh_videos/")
```

- Reads full 134 GB stream but only writes ~42 GB of val videos to disk
- After extraction, delete tarball: `rm cache/qvhilights_videos.tar.gz` (frees 134 GB)

### Step 3: Process in batches through Kairos pipeline

Modify `run_qvhighlights_benchmark.py` to support batch processing:

Add `--batch-size` and `--batch-offset` CLI args:
```bash
# Batch 1: videos 0-99
python run_qvhighlights_benchmark.py --max-videos 1519 --batch-size 100 --batch-offset 0

# Batch 2: videos 100-199
python run_qvhighlights_benchmark.py --max-videos 1519 --batch-size 100 --batch-offset 100

# ... etc
```

Each batch:
1. Loads all annotations
2. Finds videos with existing cached pipeline outputs (skips those)
3. Processes only new videos in the current batch range
4. Saves per-batch predictions JSONL
5. Pipeline outputs stay in `cache/qvhighlights_outputs/` (persistent, ~1.8 MB/video)

**Alternative (simpler):** Since pipeline outputs are cached via `--skip-pipeline`, we can:
1. Run once without `--skip-pipeline` to process all videos (long, ~60 hours for 1,519 videos at ~140s each)
2. Or run in tmux/nohup and let it run overnight across multiple sessions

**Realistic timeline:** 1,519 videos × ~140s/video = ~59 hours total pipeline time. At 100 videos/batch, that's ~3.9 hours per batch, ~15 batches.

### Step 4: Aggregate all results

After ALL batches complete, run a final aggregation pass:

```bash
python run_qvhighlights_benchmark.py --max-videos 1519 --skip-pipeline \
       --merge-adjacent --merge-gap-sec 5.0 --top-k 5
```

This uses `--skip-pipeline` so it only loads cached outputs and evaluates all queries. The official eval produces the final metrics on the full val set.

**Deduplication:** The benchmark runner indexes videos by `video_{i:03d}` where `i` is the position in the sorted video ID list. As long as we use the same `--max-videos` and annotations, the same video always gets the same index → same output directory. No duplicates possible.

### Step 5: Clean up videos (optional)

After all pipeline outputs are cached:
```bash
rm -rf test/benchmarks/cache/qvh_videos/  # ~42 GB
```

Pipeline outputs (`cache/qvhighlights_outputs/`) are small (~2.7 GB for all 1,519 videos) and contain everything needed for re-evaluation.

---

## Code Changes Required

### 1. Add selective tarball extraction to `qvhighlights_loader.py`

New function `extract_val_videos_from_tarball()`:
- Takes tarball path and val vid set
- Extracts only matching .mp4 files
- Handles nested directory structure in tarball
- Reports progress

### 2. Simplify benchmark runner for MR-only evaluation

In `run_qvhighlights_benchmark.py`:
- Keep `pred_relevant_windows` in official predictions (for MR mAP)
- Make `pred_saliency_scores` optional (skip HD eval if not needed)
- Add `--mr-only` flag to skip highlight detection computation
- Keep the `--skip-pipeline` flag for re-evaluation after batch processing

### 3. Add batch support to CLI

New args:
- `--batch-size N` — process N videos per run (default: all)
- `--batch-offset M` — start from video index M (default: 0)
- `--download-tarball` — download and extract tarball before processing

### 4. Final report generation

Update `generate_report()` to produce a clean MR-only comparison table:

```
| Method | Training | R1@0.5 | R1@0.7 | mAP@0.5 | mAP@0.75 | mAP Avg |
|--------|----------|--------|--------|---------|----------|---------|
| Moment-DETR | Supervised | 52.89 | 33.02 | 35.65 | 17.04 | 25.49 |
| QD-DETR | Supervised | 62.40 | 44.98 | — | — | 41.22 |
| UniVTG | Supervised | 58.86 | 40.86 | — | — | 35.47 |
| **Kairos** | **Zero-shot** | **??** | **??** | **??** | **??** | **??** |
```

---

## Files Modified

| File | Changes |
|------|---------|
| `test/benchmarks/dataload/qvhighlights_loader.py` | Add `extract_val_videos_from_tarball()` function |
| `test/benchmarks/run_qvhighlights_benchmark.py` | Add `--mr-only`, `--batch-size`, `--batch-offset`, `--download-tarball` flags; make HD eval optional |

No changes to `src/` (constraint maintained).

---

## Execution Plan (Step-by-Step Commands)

```bash
# 1. Download tarball (~134 GB, run in tmux)
cd /home/Kairos_model/test/benchmarks/cache
wget -c "https://nlp.cs.unc.edu/data/jielei/qvh/qvhilights_videos.tar.gz"

# 2. Extract only val videos
python -c "
import tarfile, json
val_vids = set()
with open('highlight_val_release.jsonl') as f:
    for line in f:
        val_vids.add(json.loads(line)['vid'])
print(f'Extracting {len(val_vids)} val videos...')
count = 0
with tarfile.open('qvhilights_videos.tar.gz', 'r:gz') as tar:
    for member in tar:
        name = member.name.split('/')[-1].replace('.mp4', '')
        if name in val_vids:
            member.name = member.name.split('/')[-1]
            tar.extract(member, path='qvh_videos/')
            count += 1
            if count % 100 == 0:
                print(f'  Extracted {count}/{len(val_vids)}')
print(f'Done. Extracted {count} videos.')
"

# 3. Delete tarball to free space
rm qvhilights_videos.tar.gz

# 4. Process all videos through pipeline (run in tmux, takes ~60 hours)
cd /home/Kairos_model
python test/benchmarks/run_qvhighlights_benchmark.py \
    --max-videos 1519 --mr-only --top-k 5

# 5. After all processing, re-evaluate with scene merging
python test/benchmarks/run_qvhighlights_benchmark.py \
    --max-videos 1519 --skip-pipeline --mr-only \
    --merge-adjacent --merge-gap-sec 5.0 --top-k 5

# 6. Clean up video files (keep pipeline outputs)
rm -rf test/benchmarks/cache/qvh_videos/
```

---

## Verification

1. After extraction: `ls cache/qvh_videos/*.mp4 | wc -l` should show ~1,519
2. After full pipeline: `ls cache/qvhighlights_outputs/ | wc -l` should show ~1,519
3. Final eval should report `num_queries: 1550` (matching full val set)
4. Compare R1@0.5 against Moment-DETR's 52.89 — this is the primary benchmark number
5. Predictions JSONL can be independently verified by running `python metrics/standalone_eval/eval.py --submission_path <pred.jsonl> --gt_path cache/highlight_val_release.jsonl --save_path results/official_eval.json`

---

## Timeline Estimate

| Step | Duration | Disk impact |
|------|----------|-------------|
| Download tarball | 30 min - 3 hrs | +134 GB |
| Extract val videos | 30 min - 1 hr | +42 GB |
| Delete tarball | instant | -134 GB |
| Pipeline (1,519 videos) | ~60 hours | +2.7 GB |
| Final evaluation | ~5 min | — |
| Delete videos (optional) | instant | -42 GB |

**Total wall time: ~3 days** (dominated by pipeline processing at ~140s/video).
Can be parallelized across multiple machines if available.
