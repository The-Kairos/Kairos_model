# Understanding Clip Retrieval Metrics — A Visual Guide

This document explains every metric used in the QVHighlights clip retrieval benchmark, with diagrams and worked examples. No prior knowledge assumed.

---

## The Task

Kairos receives a **video** and a **natural-language query** (a question or description). It must return the **time window** (start and end timestamps) where that query happens in the video.

```
INPUT:   Video (150 seconds long) + Query: "Woman makes a sandwich"
OUTPUT:  Predicted clip: [115s - 119s]
ANSWER:  Ground truth:   [110s - 124s]

Question: How good was the prediction?
```

---

## Metric 1: IoU (Intersection over Union)

IoU measures how much the predicted clip overlaps with the correct answer. It is a single number between 0 and 1.

### How to Calculate IoU

**Step 1 — Find the Intersection (overlap):**

The intersection is where the two time ranges overlap. Take the later start and the earlier end.

```
Intersection start = max(predicted_start, gt_start)
Intersection end   = min(predicted_end, gt_end)
Intersection       = max(0, intersection_end - intersection_start)
```

**Step 2 — Find the Union (total span covered):**

The union is the total time covered by either range. Take the earlier start and the later end.

```
Union = max(predicted_end, gt_end) - min(predicted_start, gt_start)
```

**Step 3 — Divide:**

```
IoU = Intersection / Union
```

### Worked Example 1: Decent Overlap

```
Timeline (seconds):  100  105  110  115  120  125  130
                      |    |    |    |    |    |    |
GT (correct):              |==================|
                          110                 124

Prediction:                     |=========|
                               115       122

Intersection:                   |=========|
                               115       122    = 122 - 115 = 7 seconds

Union:                    |==================|
                         110                124   = 124 - 110 = 14 seconds

IoU = 7 / 14 = 0.50
```

This means 50% overlap. At the IoU=0.5 threshold, this counts as a HIT.

### Worked Example 2: Perfect Match

```
Timeline (seconds):  120  125  130  135  140
                      |    |    |    |    |
GT (correct):              |=========|
                          122       138

Prediction:                |=========|
                          121       137

Intersection:              |=========|
                          122       137    = 137 - 122 = 15 seconds

Union:                    |==========|
                         121        138    = 138 - 121 = 17 seconds

IoU = 15 / 17 = 0.88
```

Near-perfect. Kairos nailed it.

### Worked Example 3: Right Spot, Too Narrow (Our Main Problem)

```
Timeline (seconds):  60   70   80   90   100
                      |    |    |    |    |
GT (correct):         |==========================|
                     60                          92

Prediction:                         |===|
                                   85  87

Intersection:                       |===|
                                   85  87     = 87 - 85 = 2 seconds

Union:                |==========================|
                     60                          92  = 92 - 60 = 32 seconds

IoU = 2 / 32 = 0.0625
```

Kairos found the right part of the video (85-87 is inside 60-92) but returned only a 2-second scene within a 32-second ground truth. The IoU is very low even though the prediction is in the correct location. **This is the scene granularity problem** — Kairos scenes are short, GT windows are wide.

### Worked Example 4: Complete Miss

```
Timeline (seconds):  0    20   40   60   80   100
                      |    |    |    |    |    |
GT (correct):         |=======|
                     0        28

Prediction:                              |=========|
                                        87       102

Intersection:        (no overlap at all)  = 0 seconds

Union:               |=================================|
                    0                                102  = 102 seconds

IoU = 0 / 102 = 0.00
```

Completely wrong part of the video. Zero overlap.

### Worked Example 5: Prediction Too Wide

```
Timeline (seconds):  0    30   60   90   120
                      |    |    |    |    |
GT (correct):                   |===|
                                66  90

Prediction:          |===========================|
                    0                           124

Intersection:                   |===|
                                66  90     = 90 - 66 = 24 seconds

Union:               |===========================|
                    0                           124  = 124 seconds

IoU = 24 / 124 = 0.19
```

The prediction covers the correct region but also covers everything else. IoU penalizes predictions that are too wide, not just too narrow.

### IoU Summary Table

| IoU Value | Meaning | Visual |
|-----------|---------|--------|
| 1.0 | Perfect — prediction exactly matches ground truth | `GT: \|====\|` / `PR: \|====\|` |
| 0.7+ | Excellent — strong overlap | `GT: \|========\|` / `PR: \|=======\|` |
| 0.5 | Decent — about half overlap | `GT: \|========\|` / `PR:     \|========\|` |
| 0.3 | Weak — some overlap, but mostly off | `GT: \|====\|` / `PR:   \|=\|` |
| 0.0 | Miss — no overlap at all | `GT: \|====\|` / `PR:              \|====\|` |

---

## Metric 2: R@K at IoU=T (Recall at K)

This answers: **"How often does Kairos get it right?"**

### Breaking Down the Name

**R@1 IoU=0.5** has three parts:

1. **R** = Recall = "did you find it?"
2. **@1** = only look at the top-1 prediction (the single best guess)
3. **IoU=0.5** = "right" means at least 50% overlap

So **R@1 IoU=0.5 = 25%** means:

```
Out of 16 queries:
  - 4 queries: top-1 prediction had IoU >= 0.5  (HIT)
  - 12 queries: top-1 prediction had IoU < 0.5  (MISS)

R@1 IoU=0.5 = 4 / 16 = 25%
```

### R@1 vs R@5

Kairos retrieves multiple candidate clips, ranked by similarity score. R@K checks the top K candidates:

```
Query: "Woman makes a sandwich"

Kairos returns (ranked by similarity):
  #1: [115s - 119s]  IoU = 0.25  ← R@1 checks only this one
  #2: [108s - 112s]  IoU = 0.14
  #3: [45s  - 52s]   IoU = 0.00
  #4: [110s - 120s]  IoU = 0.64  ← R@5 finds this one!
  #5: [88s  - 95s]   IoU = 0.00

R@1 IoU=0.5 → MISS (best in top-1 is 0.25, which is < 0.5)
R@5 IoU=0.5 → HIT  (best in top-5 is 0.64, which is >= 0.5)
```

This is why our R@5 (50%) is much higher than R@1 (25%) — the correct scene is often in the top-5 but not always ranked first.

### How to Calculate R@K IoU=T

```
For each query:
  1. Take the top-K predicted clips
  2. Compute IoU of each against all ground-truth windows
  3. Take the best IoU found among the top-K clips
  4. If best IoU >= T: count as HIT
  5. If best IoU < T:  count as MISS

R@K IoU=T = (number of HITs) / (total queries) × 100%
```

### Worked Example: Computing R@1 IoU=0.5 Across 4 Queries

```
Query 1: top-1 IoU = 0.72  → HIT  (0.72 >= 0.5)
Query 2: top-1 IoU = 0.10  → MISS (0.10 < 0.5)
Query 3: top-1 IoU = 0.85  → HIT  (0.85 >= 0.5)
Query 4: top-1 IoU = 0.00  → MISS (0.00 < 0.5)

R@1 IoU=0.5 = 2 HITs / 4 total = 50%
```

### All R@K IoU=T Combinations We Report

| Metric | K | IoU Threshold | Strictness | Our Score |
|--------|---|---------------|------------|-----------|
| R@1 IoU=0.3 | 1 | 0.3 (lenient) | Medium | 37.5% |
| R@1 IoU=0.5 | 1 | 0.5 (standard) | High | 25.0% |
| R@1 IoU=0.7 | 1 | 0.7 (strict) | Very High | 25.0% |
| R@5 IoU=0.3 | 5 | 0.3 (lenient) | Low | 75.0% |
| R@5 IoU=0.5 | 5 | 0.5 (standard) | Medium | 50.0% |
| R@5 IoU=0.7 | 5 | 0.7 (strict) | High | 31.2% |

**Reading this table:**
- R@5 IoU=0.3 = 75% → In 3 out of 4 queries, at least one of the top-5 clips has 30%+ overlap
- R@1 IoU=0.7 = 25% → In only 1 out of 4 queries does the single best clip overlap 70%+

---

## Metric 3: mIoU (Mean Intersection over Union)

This is the simplest metric: **average the IoU of the top-1 prediction across all queries.**

### How to Calculate

```
Query 1: top-1 IoU = 0.72
Query 2: top-1 IoU = 0.10
Query 3: top-1 IoU = 0.85
Query 4: top-1 IoU = 0.00    ← complete miss, drags the average down

mIoU = (0.72 + 0.10 + 0.85 + 0.00) / 4 = 0.4175
```

### Why Total Misses "Drag It Down"

Without the miss: (0.72 + 0.10 + 0.85) / 3 = 0.557
With the miss:    (0.72 + 0.10 + 0.85 + 0.00) / 4 = 0.418

One query with IoU=0.0 dropped the average from 0.557 to 0.418. When you have many total misses (like our IoU=0.000 on the enamel pin query), they pull mIoU down significantly even if the other predictions are decent.

### Our mIoU Breakdown

```
All 16 query IoUs:

0.300  (near-miss, right spot but too narrow)
0.071  (miss, too narrow)
0.719  ★ HIT
0.075  (miss, too narrow)
0.227  (miss, prediction too wide)
0.058  (miss, too narrow)
0.361  (near-miss)
0.039  (miss, too narrow)
0.798  ★ HIT
0.143  (miss, too narrow)
0.136  (miss, too narrow)
0.248  (miss, too narrow)
0.000  ← total miss (wrong part of video entirely)
0.855  ★ HIT
0.132  (miss, too narrow)
0.906  ★ HIT

Sum   = 5.068
Count = 16
mIoU  = 5.068 / 16 = 0.317
```

Remove the 4 best and 1 worst, and the remaining 11 queries average IoU = 0.163 — mostly caused by the scene granularity problem.

---

## How the Metrics Connect

```
                    ┌──────────────────────────────────────┐
                    │         For each query:              │
                    │                                      │
                    │  1. Kairos returns top-K clips       │
                    │  2. Compute IoU for each clip        │
                    │     against ground truth              │
                    │                                      │
                    ├──────────────────────────────────────┤
                    │                                      │
                    │  IoU of top-1 clip ──┬── mIoU        │
                    │  (one number)        │  (average     │
                    │                      │   across all  │
                    │                      │   queries)    │
                    │                      │               │
                    │  Best IoU in ────────┼── R@K IoU=T   │
                    │  top-K clips         │  (% of queries│
                    │                      │   where best  │
                    │                      │   IoU >= T)   │
                    └──────────────────────────────────────┘
```

- **IoU** = per-query, per-clip measure of overlap
- **mIoU** = average IoU of the top-1 clip across all queries (one number for the whole benchmark)
- **R@K IoU=T** = what percentage of queries had at least one good clip in the top-K

---

## What Our Numbers Tell Us About Kairos

```
R@1 IoU=0.5 = 25%     Kairos's #1 pick is correct 1 in 4 times
R@5 IoU=0.5 = 50%     Correct answer is in top-5 half the time
R@5 IoU=0.3 = 75%     With lenient overlap, top-5 covers 3 in 4 queries
mIoU        = 0.317   Average overlap is ~32% (dragged down by misses)
```

**Translation:** Kairos understands what's in the video and usually finds the right region, but it returns clips that are too short compared to what QVHighlights considers the "correct" answer. The retrieval ranking is decent (correct answer in top-5 most of the time) but needs better scene merging to produce wider clips that match the ground-truth annotation style.
