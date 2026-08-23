# Kairos QVHighlights Results — Current State

**Benchmark run date:** 2026-06-28
**Split:** Test (same as paper Table 3)
**Videos evaluated:** 1,529
**Queries evaluated:** 1,542 (match_number=True, full coverage)
**Evaluation code:** Official Moment-DETR `standalone_eval` (verified function-by-function)

---

## Our Results

| Metric | Kairos (Zero-Shot) |
|--------|-------------------|
| R1@0.5 | 38.91% |
| R1@0.7 | 22.83% |
| mAP@0.5 | 36.95% |
| mAP@0.75 | 18.74% |
| mAP Avg | 20.64% |

### Performance by Ground Truth Moment Length

| Length Bucket | mAP Avg |
|--------------|---------|
| Short (0-10s) | 5.37 |
| Middle (10-30s) | 21.34 |
| Long (30-150s) | 23.06 |

Short moments are the weakness — Kairos scenes are typically 5-15 seconds, so a 4-second ground truth gets low IoU even when the right content is found.

---

## How Kairos Does Moment Retrieval

1. **Scene Segmentation** — PySceneDetect splits the 150s video into ~10-35 variable-length scenes
2. **Multimodal Description** — Each scene gets a rich text description (BLIP captions + YOLO objects + Whisper ASR + MIT AST audio + LLM fusion)
3. **Embedding** — Scene descriptions embedded via Gemini embedding model
4. **Query Retrieval** — Query embedded with same model, top-5 scenes by cosine similarity
5. **Scene Merging** — Adjacent scenes within 5s gap merged into single windows
6. **Submission** — Top-10 windows formatted for official evaluation

Key settings: `top_k=5`, `merge_gap=5.0s`, `max_pred_windows=10`.

---

## Reproducibility Status

| What | Reproducible? | Notes |
|------|--------------|-------|
| Evaluation from saved predictions | **Yes** | `--merge-results` on the committed JSONL file. Deterministic numpy math. |
| Re-running retrieval | **Partial** | Calls Gemini embedding API for each query — model version drift changes results |
| Re-running full pipeline | **No guarantee** | Requires BLIP, YOLO, Whisper, Gemini APIs. Model versions not pinned. |
| Video data | **At risk** | 134 GB tarball on UNC server, no checksum saved. Cache deleted from disk. |
| Evaluation code | **Yes** | Committed at `test/benchmarks/metrics/qvhighlights/standalone_eval/` |

### Key Files

- Results JSON: `test/benchmarks/results/qvhighlights/qvhighlights_results_MERGED_20260628_152004.json`
- Predictions JSONL: `test/benchmarks/results/qvhighlights/qvhighlights_predictions_MERGED_20260628_152004.jsonl`
- Benchmark runner: `test/benchmarks/results/qvhighlights/run_qvhighlights_benchmark.py`
- Batch script: `test/benchmarks/results/qvhighlights/run_qvh_test_benchmark.sh`
- Data loader: `test/benchmarks/dataload/qvhighlights_loader.py`
- Full analysis: `test/benchmarks/results/qvhighlights/qvhighlights_comprehensive_analysis.md`

### What's Missing from Disk

The `test/benchmarks/cache/qvhighlights/` directory is gone (OneDrive eviction or manual cleanup). This included:
- `qvhilights_videos.tar.gz` (~134 GB tarball)
- `qvh_videos/` (~35 GB extracted clips)
- `qvhighlights_test_outputs/` (~3 GB per-video pipeline outputs)
- `highlight_test_with_gt.jsonl` (ground truth annotations)

The results are committed to git and can be re-evaluated, but re-running the pipeline from scratch would require re-downloading everything.
