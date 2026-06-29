# Comprehensive Benchmarking Analysis: Kairos on QVHighlights

## Table of Contents
1. [Dataset Background](#1-dataset-background)
2. [Task Definition: Moment Retrieval](#2-task-definition-moment-retrieval)
3. [Metric Definitions](#3-metric-definitions)
4. [Evaluation Split Verification](#4-evaluation-split-verification)
5. [Evaluation Code Verification](#5-evaluation-code-verification)
6. [Baseline Methods](#6-baseline-methods)
7. [Kairos Method Description](#7-kairos-method-description)
8. [Results](#8-results)
9. [Validity Analysis](#9-validity-analysis)
10. [Conclusion](#10-conclusion)
11. [References](#11-references)

---

## 1. Dataset Background

**Full Name:** Query-based Video Highlights (QVHighlights)

**Citation:** Jie Lei, Tamara L. Berg, and Mohit Bansal. "QVHighlights: Detecting Moments and Highlights in Videos via Natural Language Queries." In *Proceedings of the 35th Conference on Neural Information Processing Systems (NeurIPS 2021)*, Sydney, Australia. arXiv:2107.09609v2.

**Source Code and Data:** https://github.com/jayleicn/moment_detr

**Dataset Overview:**
QVHighlights consists of over 10,000 YouTube videos covering a diverse range of topics. The videos are sourced from three main categories:

- **Daily vlogs** (46.5% of test queries): everyday activities such as cooking, cleaning, shopping
- **Travel vlogs** (43.1% of test queries): travel, sightseeing, hotel tours, beach activities
- **News videos** (10.4% of test queries): protests, natural disasters, press conferences, weather reports

**Video Format:**
Raw YouTube videos (5-30 minutes long, uploaded after 2016) are segmented into **150-second clips** for annotation. Each clip is further divided into **2-second segments** for detailed temporal annotation. This means each video in the dataset is exactly 150 seconds (2.5 minutes) long, containing 75 two-second clips.

**Annotation Process:**
1. **Query and moment annotation:** Amazon Mechanical Turk workers watched each 150-second video and wrote a free-form natural language query describing an interesting activity. They then selected all 2-second clips relevant to that query. Unlike previous datasets that only allow a single moment per query, QVHighlights allows **multiple disjoint moments** (average 1.8 moments per query).
2. **Saliency score annotation:** A separate set of workers rated each relevant clip on a 5-point Likert scale (Very Bad to Very Good) from 3 different annotators.

**Quality Control:**
Only Mechanical Turk workers with >500 completed HITs and >95% approval rate qualified. Workers also had to pass a 7-question qualification test (48% pass rate among 543 workers). Inter-annotator agreement was high: 90% of queries had average Intersection-over-Union (IoU) scores above 0.9 across 3 independent moment annotations.

**Dataset Statistics:**
| Statistic | Value |
|-----------|-------|
| Total queries | 10,310 |
| Total videos | 10,148 |
| Average query length (words) | 11.3 |
| Average moment length (seconds) | 24.6 |
| Average video length (seconds) | 150 |
| Average moments per query | 1.8 |
| Short moments (0-10 seconds) | ~38% |
| Long moments (>30 seconds) | ~23% |

**Data Splits:**
| Split | Queries | Videos |
|-------|---------|--------|
| Train | ~7,218 | ~7,218 |
| Validation | 1,550 | 1,550 |
| Test | 1,542 | 1,529 |

(Source: Table 3 caption and annotation file line counts from the official repository)

---

## 2. Task Definition: Moment Retrieval

**Moment Retrieval (MR)** is the task of localizing one or more temporal segments (called "moments") within a video that are relevant to a given natural language query.

**Formal Definition:**
Given a natural language query *q* and a video *v* composed of *L_v* clips, the goal is to predict a set of temporal windows {[*t_start*, *t_end*]} that correspond to the moments in the video relevant to the query.

**Key Characteristics of QVHighlights Moment Retrieval:**
- **Multiple moments per query:** Unlike earlier datasets (DiDeMo, CharadesSTA, ActivityNet Captions) that annotate only a single moment per query-video pair, QVHighlights annotates **all** relevant moments. A query like "a man walks his dog" might correspond to 3 separate segments in the video where this activity occurs.
- **Variable-length moments:** Ground truth moments range from 2 seconds to the full 150-second video length.
- **Exhaustive annotation:** All relevant moments are annotated; there are no missing positives within a video.

**What We Are NOT Evaluating:**
This benchmark evaluates only Moment Retrieval. QVHighlights also defines a **Highlight Detection (HD)** task that involves predicting per-clip saliency scores. We do not evaluate Highlight Detection because Kairos does not produce clip-level saliency scores.

---

## 3. Metric Definitions

All metrics below are computed using the official evaluation code from the Moment-DETR repository (`standalone_eval/eval.py`). We describe each metric with its precise mathematical definition.

### 3.1 Intersection over Union (IoU)

IoU measures the overlap between a predicted temporal window and a ground truth window:

```
IoU(pred, gt) = intersection(pred, gt) / union(pred, gt)

where:
  intersection = max(0, min(pred_end, gt_end) - max(pred_start, gt_start))
  union = max(pred_end, gt_end) - min(pred_start, gt_start)
```

IoU ranges from 0 (no overlap) to 1 (perfect overlap).

### 3.2 Recall@1 at IoU Threshold (R1@0.5, R1@0.7)

**Definition:** The percentage of queries for which the **top-1 ranked** predicted window has IoU greater than or equal to a threshold with the best-matching ground truth window.

**Computation (from `compute_mr_r1()` in the official code):**
1. For each query, take the single highest-scored predicted window.
2. Compute IoU between this prediction and every ground truth window for that query.
3. Select the ground truth window with the highest IoU.
4. The query is a "hit" if this IoU >= threshold.
5. R1@threshold = (number of hits) / (total number of queries) x 100

**Thresholds used:**
- **R1@0.5**: A lenient threshold. The predicted window must overlap at least 50% with the best ground truth. This measures whether the system roughly finds the right temporal region.
- **R1@0.7**: A strict threshold. The predicted window must overlap at least 70% with the best ground truth. This measures precise temporal localization.

### 3.3 Mean Average Precision (mAP@0.5, mAP@0.75, mAP Avg)

**Definition:** Average Precision (AP) is computed per-query over the ranked list of predicted windows. AP is then averaged across all queries.

**Computation (from `compute_mr_ap()` in the official code):**
1. For each query, take the top-10 predicted windows (max_pred_windows=10), ranked by score.
2. At a given IoU threshold:
   a. Walk through predictions in score order.
   b. A prediction is a true positive if its IoU with a not-yet-matched ground truth window exceeds the threshold.
   c. Each ground truth can only be matched once (greedy matching by highest IoU first).
   d. Compute interpolated precision-recall curve and integrate to get Average Precision.
3. Average AP across all queries to get mAP at that threshold.

**Thresholds used:**
- **mAP@0.5**: Average Precision at IoU >= 0.5
- **mAP@0.75**: Average Precision at IoU >= 0.75
- **mAP Avg**: Average of mAP computed at **10 IoU thresholds** evenly spaced from 0.50 to 0.95:
  `[0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]`
  This is the strictest aggregate metric, as it includes very demanding thresholds (0.90, 0.95) where predictions must be nearly exact.

### 3.4 Length Buckets

Metrics are also computed separately for subsets of queries based on ground truth moment length:

| Bucket | Ground Truth Window Length | Description |
|--------|--------------------------|-------------|
| Short | 0-10 seconds | Brief actions (e.g., "a person opens a door") |
| Middle | 10-30 seconds | Medium-length activities (e.g., "someone cooks pasta") |
| Long | 30-150 seconds | Extended activities (e.g., "a tour of the hotel lobby") |
| Full | 0-150 seconds | All queries combined |

A query is assigned to a bucket based on the length of its ground truth windows. If a query has multiple ground truth windows of different lengths, it may appear in multiple buckets (each bucket filters by individual window length, not query-level aggregation).

---

## 4. Evaluation Split Verification

### Evidence That Table 3 Uses the Test Split

**Table 3 caption (verbatim from the paper, page 8):**
> "Table 3: Baseline Comparison on QVHIGHLIGHTS **test split**. We highlight the best score in each column in bold, and the second best score with underline."

**Paper text immediately following Table 3 (page 8, Section 5.2):**
> "We compare Moment-DETR with various moment retrieval and highlight detection methods on the QVHIGHLIGHTS **test split**; results are shown in Table 3."

### Which Tables Use Which Splits

| Table | Split | Content |
|-------|-------|---------|
| Table 3 | **Test** | Main baseline comparison (MCN, CAL, CLIP, XML, XML+, Moment-DETR) |
| Table 4 | Val | Loss ablation study (Moment-DETR only) |
| Table 5 | Val | Pretraining data domain/size study (Moment-DETR only) |
| Table 7 | **Test** | Performance breakdown by video category |
| Table 8 | Val | Ablation on number of moment queries (Moment-DETR only) |
| Table 9 | Val | Saliency loss ablation (Moment-DETR only) |

### What We Used

- **Split:** Test
- **Ground truth file:** `highlight_test_with_gt.jsonl` (downloaded from the official Moment-DETR GitHub repository)
- **Number of queries evaluated:** 1,542 (matches the full test set)
- **Number of videos processed:** 1,529 (all unique test videos)
- **`match_number=True`**: The official evaluation enforced an exact match between predicted and ground truth query IDs, confirming complete coverage

---

## 5. Evaluation Code Verification

### Source of Our Evaluation Code

Our evaluation code is located at `test/benchmarks/metrics/standalone_eval/eval.py`. The file header states:

> "Adapted from Moment-DETR standalone_eval/eval.py  
> Original: https://github.com/jayleicn/moment_detr/blob/main/standalone_eval/eval.py"

The utility functions are in `test/benchmarks/metrics/standalone_eval/utils.py`, which states:

> "Adapted from Moment-DETR standalone_eval/utils.py  
> Original: https://github.com/jayleicn/moment_detr/blob/main/standalone_eval/utils.py  
> Originally from MMAction2: https://github.com/open-mmlab/mmaction2"

### Key Parameters Verified

| Parameter | Value in Paper/Official Code | Value in Our Code | Match |
|-----------|-------|---------|-------|
| IoU thresholds for mAP | np.linspace(0.5, 0.95, 10) | np.linspace(0.5, 0.95, 10) | Yes |
| max_pred_windows (mAP) | 10 | 10 | Yes |
| R1 computation | Top-1 prediction vs best-matching GT | Same | Yes |
| Length buckets | [0,10], [10,30], [30,150], [0,150] | [0,10], [10,30], [30,150], [0,150] | Yes |
| AP computation | Interpolated precision-recall | Interpolated precision-recall | Yes |
| match_number | True (for final results) | True | Yes |

### Functions Verified

| Function | Purpose | Status |
|----------|---------|--------|
| `eval_submission()` | Entry point; dispatches to MR and HD eval | Verified identical logic |
| `eval_moment_retrieval()` | Iterates over length buckets | Verified identical |
| `compute_mr_ap()` | Computes mAP with greedy GT matching | Verified identical |
| `compute_mr_r1()` | Computes R1 at IoU thresholds | Verified identical |
| `compute_average_precision_detection()` | Per-query AP with interpolated PR curve | Verified identical |
| `compute_temporal_iou_batch_cross()` | Cross-product IoU computation | Verified identical |
| `compute_temporal_iou_batch_paired()` | Paired IoU computation | Verified identical |

---

## 6. Baseline Methods

All baseline numbers in this section are taken directly from **Table 3 of Lei et al. (2021), evaluated on the QVHighlights test split.** Reference numbers in square brackets (e.g., [13]) refer to the paper's bibliography.

### 6.1 MCN — Moment Context Network

**Full Citation:** Lisa Anne Hendricks, Oliver Wang, Eli Shechtman, Josef Sivic, Trevor Darrell, and Bryan Russell. "Localizing Moments in Video with Natural Language." In *International Conference on Computer Vision (ICCV)*, 2017. [13]

**Method:**
MCN is a **proposal-based** moment retrieval method. It generates candidate temporal proposals (video segments of varying lengths), then scores each proposal by comparing its visual features against text query features. The model considers not just the proposal itself but also its temporal context (what happens before and after the proposed moment).

**Training:** Supervised on the QVHighlights training split with ground truth moment annotations.

**Results (Table 3, test split):**
| R1@0.5 | R1@0.7 | mAP@0.5 | mAP@0.75 | mAP Avg |
|--------|--------|---------|----------|---------|
| 11.41 | 2.72 | 24.94 | 8.22 | 10.67 |

---

### 6.2 CAL — Context-Aware Localization

**Full Citation:** Victor Escorcia, Mattia Soldan, Josef Sivic, Bernard Ghanem, and Bryan Russell. "Temporal Localization of Moments in Video Collections with Natural Language." arXiv preprint arXiv:1907.12763, 2019. [4]

**Method:**
CAL is another **proposal-based** method that improves upon MCN by incorporating richer temporal context around candidate moment proposals. It explicitly models the relationship between a candidate moment and its surrounding video content to better discriminate relevant moments.

**Training:** Supervised on the QVHighlights training split with ground truth moment annotations.

**Results (Table 3, test split):**
| R1@0.5 | R1@0.7 | mAP@0.5 | mAP@0.75 | mAP Avg |
|--------|--------|---------|----------|---------|
| 25.49 | 11.54 | 23.40 | 7.65 | 9.89 |

---

### 6.3 CLIP — Contrastive Language-Image Pre-training

**Full Citation:** Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. "Learning Transferable Visual Models from Natural Language Supervision." arXiv preprint arXiv:2103.00020, 2021. [31]

**Method:**
CLIP is used as a **zero-shot baseline** in the QVHighlights paper. It was NOT trained on any QVHighlights data. The paper describes its application to moment retrieval as follows (page 8, Section 5.2):

> "We compute clip-wise similarity scores by computing image-query scores where the image is the center frame of the clip. For moment retrieval, we use TAG [50] to progressively groups top-scored clips with the classical watershed algorithm [35]."

Step by step:
1. For each 2-second clip in the video, extract the **center frame** (a single image).
2. Encode this frame using the CLIP image encoder (Vision Transformer, ViT-B/32 architecture).
3. Encode the text query using the CLIP text encoder.
4. Compute the **cosine similarity** between the image embedding and the text embedding for each clip.
5. Use **TAG (Temporal Adjacent Grouping)** [50] with the **watershed algorithm** [35] to progressively merge adjacent high-scoring clips into contiguous moment windows.

**TAG** (Zhao et al., "Bottom-Up Temporal Action Detection with Mutual Regularization", ECCV 2020 [50]) is a temporal grouping algorithm that starts from peak-scoring clips and expands outward, merging adjacent clips as long as their scores remain above a threshold. The **watershed algorithm** [35] (Roerdink and Meijster, "The Watershed Transform: Definitions, Algorithms and Parallelization Strategies", 2000) is a classical image segmentation technique adapted here for 1D temporal segmentation.

**Training:** None on QVHighlights. CLIP was pre-trained on 400 million image-text pairs from the internet, but this pre-training is completely independent of the QVHighlights dataset and task.

**Results (Table 3, test split):**
| R1@0.5 | R1@0.7 | mAP@0.5 | mAP@0.75 | mAP Avg |
|--------|--------|---------|----------|---------|
| 16.88 | 5.19 | 18.11 | 7.00 | 7.67 |

---

### 6.4 XML — Cross-Modal Matching

**Full Citation:** Jie Lei, Licheng Yu, Tamara L. Berg, and Mohit Bansal. "TVR: A Large-Scale Dataset for Video-Subtitle Moment Retrieval." In *European Conference on Computer Vision (ECCV)*, 2020. [19]

**Method:**
XML is a **span prediction method** (as opposed to proposal-based). Rather than generating candidate proposals and scoring them, XML directly predicts clip-wise similarity scores between video clips and the text query. These scores are used to identify the start and end of relevant moments. The paper describes XML as having "a smaller capacity than Moment-DETR" in its original form.

**Training:** Supervised on the QVHighlights training split with ground truth moment annotations.

**Results (Table 3, test split):**
| R1@0.5 | R1@0.7 | mAP@0.5 | mAP@0.75 | mAP Avg |
|--------|--------|---------|----------|---------|
| 41.83 | 30.35 | 44.63 | 31.73 | 32.14 |

---

### 6.5 XML+ — Enhanced Cross-Modal Matching

**Full Citation:** Same as XML [19], with modifications described in Lei et al. (2021) Section 5.2.

**Method:**
XML+ is an enhanced version of XML created by the QVHighlights paper authors for a fairer comparison with Moment-DETR. The paper states (page 8):

> "The original XML model has a smaller capacity than Moment-DETR, hence for a fair comparison, we increased its capacity by adding more layers and train it for the same number of epochs as Moment-DETR. Moreover, to leverage the saliency annotations in QVHighlights, we further added an auxiliary saliency loss to it (referred to as 'XML+')."

**Training:** Supervised on the QVHighlights training split, with the same number of training epochs as Moment-DETR, plus an auxiliary saliency loss leveraging saliency annotations.

**Results (Table 3, test split):**
| R1@0.5 | R1@0.7 | mAP@0.5 | mAP@0.75 | mAP Avg |
|--------|--------|---------|----------|---------|
| 46.69 | 33.46 | 47.89 | 34.67 | 34.90 |

---

### 6.6 Moment-DETR — Moment Detection Transformer

**Full Citation:** Jie Lei, Tamara L. Berg, and Mohit Bansal. "QVHighlights: Detecting Moments and Highlights in Videos via Natural Language Queries." NeurIPS 2021. [The paper itself]

**Method:**
Moment-DETR is the paper's primary contribution. It adapts the DETR (Detection Transformer) architecture — originally designed for 2D object detection by Carion et al. (ECCV 2020) [3] — for 1D temporal moment retrieval. The architecture consists of:

1. **Feature extraction:** SlowFast [5] and CLIP ViT-B/32 video features extracted every 2 seconds (producing 2816-dimensional features per clip). CLIP text encoder for query features (512-dimensional per token).
2. **Transformer encoder:** Processes the concatenated video and text features with self-attention.
3. **Transformer decoder:** Uses 10 learnable "moment queries" (positional embeddings) that attend to the encoder output and predict moment coordinates.
4. **Prediction heads:** Three separate heads predict (a) normalized moment center and width, (b) foreground/background classification, and (c) clip-wise saliency scores.
5. **Training losses:** L1 + generalized IoU loss for moment coordinates, cross-entropy for classification, hinge loss for saliency. Hungarian bipartite matching assigns predictions to ground truth moments.

**Training:** Supervised on the QVHighlights training split with full moment-level and saliency-level annotations. Results reported are the mean of 5 runs with different random seeds (standard deviations shown as +- in Table 3).

**Results (Table 3, test split):**
| R1@0.5 | R1@0.7 | mAP@0.5 | mAP@0.75 | mAP Avg |
|--------|--------|---------|----------|---------|
| 52.89 (+-2.3) | 33.02 (+-1.7) | 54.82 (+-1.7) | 29.40 (+-1.7) | 30.73 (+-1.4) |

**Note:** The paper also reports Moment-DETR with weakly supervised pretraining (Moment-DETR w/ PT) using Automatic Speech Recognition (ASR) captions, which achieves even higher scores (R1@0.5=59.78). We compare against the base Moment-DETR without pretraining.

---

## 7. Kairos Method Description

### 7.1 Overview

Kairos is a **zero-shot** video understanding system. It was NOT trained on QVHighlights or any moment retrieval dataset. It is designed as a general-purpose video scene retrieval system that segments videos into semantically coherent scenes and enables natural language search over them.

### 7.2 Pipeline

The Kairos moment retrieval pipeline operates as follows:

**Step 1 — Scene Segmentation:**
The input video is segmented into scenes using visual and audio cues. Unlike QVHighlights' fixed 2-second clips, Kairos produces **variable-length scenes** whose boundaries are determined by content changes (visual transitions, audio shifts, topic changes). A typical 150-second QVHighlights video produces approximately 10-35 scenes.

**Step 2 — Multimodal Description Generation:**
For each scene, Kairos generates a rich textual description that captures visual content, audio content, activities, and contextual information. This is fundamentally different from the CLIP baseline, which only looks at a single center frame per 2-second clip.

**Step 3 — Embedding:**
Each scene description is embedded into a vector using the Gemini embedding model (via Google's Generative AI API). This produces a dense vector representation of each scene's semantic content.

**Step 4 — Query Embedding and Retrieval:**
The user's natural language query is embedded using the same Gemini embedding model. Cosine similarity is computed between the query embedding and each scene embedding. The top-K scenes (K=5) with the highest cosine similarity are returned as predicted moments.

**Step 5 — Adjacent Scene Merging:**
If two or more retrieved scenes are temporally adjacent (within a 5-second gap), they are merged into a single contiguous moment window. The merged window's score is the maximum score among its constituent scenes.

**Step 6 — Prediction Formatting:**
The top-10 predicted windows (after merging) are formatted with [start_seconds, end_seconds, score] and submitted for evaluation using the official evaluation code.

### 7.3 Key Settings

| Parameter | Value |
|-----------|-------|
| top_k (scenes retrieved per query) | 5 |
| merge_adjacent | True |
| merge_gap_sec | 5.0 seconds |
| max_pred_windows (for mAP evaluation) | 10 |
| Embedding model | Gemini (Google Generative AI) |
| Training on QVHighlights | None (zero-shot) |

### 7.4 Key Differences from the CLIP Baseline

| Aspect | CLIP Baseline | Kairos |
|--------|--------------|--------|
| Training on QVHighlights | None (zero-shot) | None (zero-shot) |
| Video representation | Center frame of each 2-second clip | Full multimodal scene descriptions |
| Temporal segmentation | Fixed 2-second clips (75 per video) | Variable-length scenes (~10-35 per video) |
| Feature extraction | CLIP ViT-B/32 image encoder | Gemini embedding of generated text descriptions |
| Query encoding | CLIP text encoder | Gemini embedding of query text |
| Similarity computation | CLIP image-text cosine similarity | Text-text cosine similarity (descriptions vs. query) |
| Temporal grouping | TAG watershed algorithm | Adjacent scene merging (5s gap threshold) |
| Modalities used | Single visual frame | Visual + audio + contextual understanding |

---

## 8. Results

### 8.1 Full Comparison Table

All baseline numbers are from Table 3 of Lei et al. (2021), evaluated on the QVHighlights **test split**. Kairos results are from our evaluation using the same test split, same ground truth annotations, and same official evaluation code.

| Method | Training Data | R1@0.5 | R1@0.7 | mAP@0.5 | mAP@0.75 | mAP Avg |
|--------|--------------|--------|--------|---------|----------|---------|
| MCN [13] | QVH train (supervised) | 11.41 | 2.72 | 24.94 | 8.22 | 10.67 |
| CAL [4] | QVH train (supervised) | 25.49 | 11.54 | 23.40 | 7.65 | 9.89 |
| CLIP [31] | None (zero-shot) | 16.88 | 5.19 | 18.11 | 7.00 | 7.67 |
| XML [19] | QVH train (supervised) | 41.83 | 30.35 | 44.63 | 31.73 | 32.14 |
| XML+ | QVH train (supervised) | 46.69 | 33.46 | 47.89 | 34.67 | 34.90 |
| Moment-DETR | QVH train (supervised) | 52.89 | 33.02 | 54.82 | 29.40 | 30.73 |
| Moment-DETR w/ PT | QVH train + ASR PT | 59.78 | 40.33 | 60.51 | 35.36 | 36.14 |
| **Kairos** | **None (zero-shot)** | **38.91** | **22.83** | **36.95** | **18.74** | **20.64** |

### 8.2 Kairos Results by Ground Truth Moment Length

| Length Bucket | mAP Avg |
|--------------|---------|
| Short (0-10 seconds) | 5.37 |
| Middle (10-30 seconds) | 21.34 |
| Long (30-150 seconds) | 23.06 |
| Full (all) | 20.64 |

Kairos performs best on long moments and worst on short moments. This is expected: Kairos retrieves scenes (which are typically 5-15 seconds long), so very short ground truth moments (0-10 seconds) are harder to localize precisely. A retrieved scene that spans 12 seconds will have low IoU with a 4-second ground truth, even if it contains the correct content.

### 8.3 Kairos Evaluation Details

| Detail | Value |
|--------|-------|
| Results file | `qvhighlights_results_MERGED_20260628_152004.json` |
| Predictions file | `qvhighlights_predictions_MERGED_20260628_152004.jsonl` |
| Total predictions | 1,542 |
| Total ground truth queries | 1,542 |
| match_number | True (exact query ID match verified) |
| Date of evaluation | 2026-06-28 |

---

## 9. Validity Analysis

### 9.1 Is the Evaluation Methodologically Sound?

We assess the validity of our benchmark against five criteria:

#### (a) Evaluation Split: VALID

We evaluated on the **test split**, the same split used for Table 3 in the paper. This is verified by:
- The Table 3 caption explicitly states "test split"
- The paper text on page 8 confirms "QVHIGHLIGHTS test split"
- We used `highlight_test_with_gt.jsonl` containing 1,542 queries
- All 1,542 queries were evaluated (match_number=True)

#### (b) Evaluation Metrics: VALID

We used the **identical official evaluation code** from the Moment-DETR repository. Our copy at `test/benchmarks/metrics/standalone_eval/eval.py` matches the original in all key functions:
- Same IoU computation
- Same mAP computation (max_pred_windows=10, 10 IoU thresholds from 0.5 to 0.95)
- Same R1 computation (top-1 prediction vs best-matching ground truth)
- Same length bucket definitions
- Same interpolated precision-recall computation

#### (c) Ground Truth Annotations: VALID

We used the official `highlight_test_with_gt.jsonl` file from the Moment-DETR repository. This is the same ground truth used to evaluate all baselines in Table 3.

#### (d) Full Coverage: VALID

1,542 out of 1,542 test queries were evaluated, with `match_number=True` enforcing exact correspondence. No queries were dropped or missing.

#### (e) Prediction Format: VALID

Kairos predictions are formatted identically to the official submission format:
```json
{
    "qid": <query_id>,
    "query": "<query text>",
    "vid": "<video_id>",
    "pred_relevant_windows": [[start, end, score], ...]
}
```

### 9.2 Is the Comparison with CLIP Fair?

**YES — this is the most meaningful comparison.**

Both Kairos and CLIP are **zero-shot** methods. Neither was trained on any QVHighlights data. The comparison is fair because:

1. **Same evaluation conditions:** Same test split, same ground truth, same metrics, same evaluation code.
2. **Same task framing:** Both receive a text query and must localize moments in a video they have never seen during training.
3. **No data leakage:** Neither method had access to QVHighlights annotations during any stage of development.

**Differences that do NOT invalidate the comparison (but should be disclosed):**

| Factor | CLIP | Kairos | Impact |
|--------|------|--------|--------|
| Visual representation | Single center frame per 2s clip | Full multimodal scene descriptions | Kairos has richer input; this is a legitimate architectural advantage |
| Temporal granularity | 75 fixed 2-second clips per video | ~10-35 variable-length scenes | Affects IoU computation; Kairos scenes are coarser |
| Grouping algorithm | TAG watershed | Adjacent merging (5s gap) | Different temporal aggregation strategies |
| Embedding model | CLIP ViT-B/32 + text encoder | Gemini text embedding | Different embedding spaces |
| Pre-training data | 400M image-text pairs | Gemini's training data | Both are general-purpose, neither task-specific |

These differences reflect **legitimate architectural choices**, not unfair advantages. Both systems are zero-shot video moment retrieval systems evaluated under identical conditions.

### 9.3 Is the Comparison with Supervised Methods Fair?

**VALID BUT INHERENTLY ASYMMETRIC.**

The comparison with MCN, CAL, XML, XML+, and Moment-DETR is valid in the sense that all methods are evaluated on the same test split with the same metrics. However, it is inherently asymmetric because:

1. **Supervised methods** (MCN, CAL, XML, XML+, Moment-DETR) were trained on the QVHighlights training split (~7,218 query-moment pairs). They have learned the specific distribution of moments, query styles, and video content in QVHighlights.
2. **Kairos** (and CLIP) had **zero exposure** to QVHighlights data. They generalize from their general-purpose pre-training only.

This asymmetry **favors the supervised methods** and should be clearly stated in any publication. The fact that Kairos zero-shot outperforms two supervised methods (MCN and CAL) and approaches a third (XML) is noteworthy precisely because of this disadvantage.

### 9.4 Caveats and Limitations

**Caveat 1: Temporal Granularity Mismatch**
Kairos segments videos into variable-length scenes (typically 5-15 seconds), not fixed 2-second clips. This means:
- For short ground truth moments (0-10 seconds), Kairos may predict windows that are too broad, reducing IoU.
- This is reflected in the poor short-moment performance (mAP Avg = 5.37 for short vs 23.06 for long).
- CLIP, operating on 2-second clips, has finer temporal granularity for short moments.

**Caveat 2: Different Embedding Spaces**
Kairos uses Gemini embeddings; CLIP uses OpenAI's CLIP embeddings. These are fundamentally different representation spaces. A "better" embedding model could improve either system's performance.

**Caveat 3: Multimodal Richness**
Kairos generates text descriptions that capture audio, visual, and contextual information. CLIP only uses visual information (single frames). This gives Kairos a richer representation but also means it depends on the quality of its description generation, which can introduce errors.

**Caveat 4: Scene Merging Heuristic**
Kairos merges adjacent scenes within a 5-second gap. This heuristic was not extensively tuned on QVHighlights and may not be optimal. Different gap thresholds could yield different results.

**Caveat 5: Computational Cost**
Kairos requires running a full video processing pipeline (scene segmentation, description generation, embedding) for each video, which is substantially more expensive than CLIP's simple frame extraction and encoding. This is not reflected in the accuracy metrics.

---

## 10. Conclusion

### Summary of Findings

1. **The benchmark is valid.** We evaluated Kairos on the exact same test split, with the exact same metrics and evaluation code, as all baselines reported in Table 3 of the QVHighlights paper (Lei et al., NeurIPS 2021).

2. **Kairos substantially outperforms the only other zero-shot baseline (CLIP):**
   - 2.3x better on R1@0.5 (38.91% vs 16.88%)
   - 4.4x better on R1@0.7 (22.83% vs 5.19%)
   - 2.0x better on mAP@0.5 (36.95% vs 18.11%)
   - 2.7x better on mAP@0.75 (18.74% vs 7.00%)
   - 2.7x better on mAP Avg (20.64% vs 7.67%)

3. **Kairos zero-shot surpasses two supervised methods** (MCN and CAL) on all metrics.

4. **Kairos zero-shot does not surpass any method trained specifically on QVHighlights** with more than minimal capacity (XML, XML+, Moment-DETR all outperform Kairos). This is expected and does not diminish the result — zero-shot systems are not expected to match fully supervised ones.

5. **The most meaningful comparison is Kairos vs CLIP**, as both are zero-shot. Kairos's advantage comes from richer multimodal scene understanding rather than single-frame visual matching.

### Validity Statement

This benchmark evaluation is suitable for publication. The comparison uses the official evaluation protocol, the same data split, and the same evaluation code as the original paper. All baseline numbers are taken directly from the published results in Table 3 of Lei et al. (2021). The `match_number=True` flag confirms that predictions cover the full test set without gaps or duplicates.

---

## 11. References

[4] Victor Escorcia, Mattia Soldan, Josef Sivic, Bernard Ghanem, and Bryan Russell. "Temporal Localization of Moments in Video Collections with Natural Language." arXiv preprint arXiv:1907.12763, 2019.

[13] Lisa Anne Hendricks, Oliver Wang, Eli Shechtman, Josef Sivic, Trevor Darrell, and Bryan Russell. "Localizing Moments in Video with Natural Language." In ICCV, 2017.

[19] Jie Lei, Licheng Yu, Tamara L. Berg, and Mohit Bansal. "TVR: A Large-Scale Dataset for Video-Subtitle Moment Retrieval." In ECCV, 2020.

[31] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. "Learning Transferable Visual Models from Natural Language Supervision." arXiv preprint arXiv:2103.00020, 2021.

[35] Jos B.T.M. Roerdink and Arnold Meijster. "The Watershed Transform: Definitions, Algorithms and Parallelization Strategies." Fundamenta Informaticae, 41(1-2):187-228, 2000.

[50] Peize Zhao, Lingxi Xie, Chen Ju, Ya Zhang, Yanfeng Wang, and Qi Tian. "Bottom-Up Temporal Action Detection with Mutual Regularization." In ECCV, 2020.

Lei et al. (2021) — Jie Lei, Tamara L. Berg, and Mohit Bansal. "QVHighlights: Detecting Moments and Highlights in Videos via Natural Language Queries." In NeurIPS, 2021. arXiv:2107.09609.

---

*Report generated: 2026-06-28. All baseline figures verified against the text extraction of arXiv:2107.09609v2.*
