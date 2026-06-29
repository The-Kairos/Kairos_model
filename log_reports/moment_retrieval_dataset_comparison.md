# Moment Retrieval Dataset Comparison — Which Benchmark Fits Kairos?

**Date:** 2026-06-21
**Context:** Kairos uses PySceneDetect to segment videos into scenes, then retrieves clips via embedding similarity. We need a benchmark dataset that evaluates this clip retrieval capability with natural language queries and ground-truth temporal windows.

---

## The Question

We ran QVHighlights and got promising results (R@1 IoU=0.5 = 50% with scene merging, approaching supervised baselines). But QVHighlights videos are only **2.5 minutes** long — Kairos is designed for **long-form video understanding**. Should we:

1. **Stay with QVHighlights** — improve scores via scene merging and more videos?
2. **Switch to a long-video dataset** — MAD, Ego4D NLQ, or something else?

This document compares the major moment retrieval datasets to answer that question.

---

## Side-by-Side Comparison

| Feature | QVHighlights | MAD | Ego4D NLQ | Charades-STA | ActivityNet Captions | TACoS |
|---------|-------------|-----|-----------|--------------|---------------------|-------|
| **Paper** | Lei et al., NeurIPS 2021 | Soldan et al., CVPR 2022 | Grauman et al., CVPR 2022 | Gao et al., ICCV 2017 | Krishna et al., ICCV 2017 | Regneri et al., TACL 2013 |
| **Video source** | YouTube | Movies | Egocentric wearable cameras | Crowd-sourced indoor | YouTube | MPII Cooking |
| **Domain** | Diverse (vlogs, news, how-to) | Cinema (90 years, 22 genres) | First-person daily life | Indoor home activities | Diverse human activities | Cooking only |
| **Avg video length** | **~2.5 min** | **~1.85 hours** | **~8-20 min (clips), hours (full)** | **~30 sec** | **~2 min** | **~4.8 min** |
| **Total hours** | ~64 hrs | ~1,200 hrs | ~800 hrs (all episodic memory) | ~56 hrs | ~648 hrs | ~10 hrs |
| **Total videos** | 10,310 queries / 10,148 videos | 384K sentences / 650 movies | ~74K queries / 1,000+ hrs | ~16K queries / 6,672 videos | ~72K sentences / 20K videos | ~18K queries / 127 videos |
| **Val split size** | 1,550 queries / 1,519 videos | ~10K queries | ~4K queries | ~3.7K queries | ~4.9K sentences | ~4.K queries |
| **NL queries?** | Yes — free-form | Yes — audio descriptions (narrated) | Yes — template-guided free-form | Yes — free-form | Yes — dense captions | Yes — free-form |
| **GT time windows?** | Yes — [start, end] seconds | Yes — [start, end] seconds | Yes — [start, end] seconds + frames | Yes — [start, end] seconds | Yes — [start, end] seconds | Yes — [start, end] seconds |
| **Standard metrics** | R@1 IoU=0.5/0.7, mAP | R@1/5/10/50/100 | R@1/5 IoU=0.3/0.5 | R@1/5 IoU=0.5/0.7 | R@1/5 IoU=0.5/0.7 | R@1/5 IoU=0.5/0.7 |
| **Videos available?** | ~1% (YouTube rot) | NDA required, features only (no raw video) | License agreement required | Downloadable | ~60-70% (YouTube rot) | Downloadable |
| **Suitable for Kairos?** | Partially (too short) | Ideal length, hard access | Good length, egocentric domain | Too short | Medium length | Too short, narrow domain |

---

## Detailed Dataset Profiles

### 1. QVHighlights (Current Benchmark)

**What it is:** Diverse YouTube clips with human-written moment queries. Each video is a 2.5-minute excerpt from a longer YouTube video (news, vlogs, how-to, travel).

**Query style:** Free-form natural language.
- "A woman is looking out over a misty valley through some trees"
- "Woman makes herself a sandwich"
- "A black screen with texts describing events not shown in the video"

**GT windows:** Yes — `relevant_windows: [[start, end], ...]` in seconds. Multiple windows per query allowed.

**Strengths for Kairos:**
- Exact same evaluation protocol we already implemented (R@K IoU=T, mIoU)
- Diverse content matches Kairos's general-purpose design
- Published supervised baselines for comparison (Moment-DETR: 52.89%, QD-DETR: 62.40%)

**Weaknesses for Kairos:**
- Videos are only 2.5 minutes — Kairos is built for long-form video
- PySceneDetect produces 3-15 scenes per 2.5-min clip — too granular vs. GT windows (20-90s)
- Only ~1% of videos still available on YouTube (16 out of 1,519 in val)
- GT windows were annotated for short clips, so "moment retrieval" means finding a 20-90s window within a 150s video — not challenging for temporal localization

**Our results (16 videos, with scene merging):**

| Metric | Kairos (zero-shot) | Moment-DETR (supervised) |
|--------|-------------------|--------------------------|
| R@1 IoU=0.3 | 81.2% | — |
| R@1 IoU=0.5 | 50.0% | 52.89% |
| R@1 IoU=0.7 | 25.0% | 33.02% |
| mIoU | 0.549 | — |

---

### 2. MAD (Movie Audio Descriptions)

**What it is:** 384K natural language sentences grounded in 650 full-length movies (~1,200 hours). Annotations come from professional audio descriptions recorded for visually impaired viewers.

**Query style:** Descriptive narration from audio descriptions.
- "The detective walks slowly down the dimly lit corridor"
- "She picks up the phone and dials a number"

These are natural language but **not question-style queries** — they are third-person narrations of what is happening. Still, they serve the same function: given the text, find the temporal window where it occurs.

**GT windows:** Yes — each audio description sentence has a precise temporal window grounded in the movie timeline.

**Video length:** Average ~1.85 hours per movie. This is the **longest-video moment retrieval dataset** available.

**Why it's interesting for Kairos:**
- Long videos are exactly what Kairos is designed for
- PySceneDetect would produce hundreds of scenes per movie — much more realistic for scene-level retrieval
- The scene granularity problem is less severe: GT windows in MAD are typically 5-30 seconds (matching Kairos scene lengths better)
- 384K queries is massive — statistically robust evaluation
- 22 genres of cinema — diverse visual content

**Why it's problematic for Kairos:**
- **Raw movies are NOT distributed** — only pre-extracted features and annotations (NDA required)
- You must obtain the movies yourself (legally complex), OR use the provided CLIP/I3D features
- Kairos needs raw video files to run its pipeline (PySceneDetect, BLIP, YOLO, Whisper)
- Without raw video, we cannot run the Kairos pipeline — we'd need to extract features ourselves
- The NDA + movie acquisition process is slow and legally sensitive

**Access:** Google Form + NDA → email with download link + password. Movies must be sourced independently.

**Published baselines (R@1):**

| Method | R@1 | R@5 | R@10 | R@50 |
|--------|-----|-----|------|------|
| CLIP (zero-shot) | 2.2% | 5.8% | 8.6% | 23.1% |
| VLG-Net (supervised) | 3.1% | 7.3% | 10.4% | 28.5% |

Note: Numbers are low because finding a 10-second moment in a 2-hour movie is extremely difficult.

---

### 3. Ego4D NLQ (Natural Language Queries)

**What it is:** Egocentric (first-person) video from wearable cameras. Annotators wrote queries about specific moments: "When did I put the phone down?", "Where did I leave the keys?"

**Query style:** Template-guided free-form natural language. 13 templates like:
- "What did I put in [object]?"
- "Where is [object] before/after I [action]?"
- "What [object] did I [action]?"

Annotators filled templates and then wrote free-form versions, so queries are natural language.

**GT windows:** Yes — `[clip_start_sec, clip_end_sec]` and `[video_start_sec, video_end_sec]` with frame-level precision.

**Video length:** Clips are 8-20 minutes, full canonical videos can be hours long. This makes it a **medium-to-long video dataset**.

**Why it's interesting for Kairos:**
- Longer videos than QVHighlights (8-20 min clips, much more than 2.5 min)
- Natural language queries with GT windows — same evaluation protocol as QVHighlights
- Same metrics: R@1/5 IoU=0.3/0.5 — directly comparable
- Videos are downloadable (with license agreement through Ego4D consortium)
- Large scale: ~74K queries across 800+ hours

**Why it's problematic for Kairos:**
- **Egocentric (first-person) video** — Kairos's BLIP captioning, YOLO detection, and scene descriptions are tuned for third-person video. First-person video has frequent motion blur, unusual viewpoints, and hand-centric framing
- PySceneDetect may not work well on egocentric video (fewer clear shot boundaries, more continuous motion)
- The domain is narrow (daily activities) compared to Kairos's general-purpose design
- Requires signing a license agreement with the Ego4D consortium
- Download is large (~5.4 TB for full dataset)

**Access:** Sign license at ego4d-data.org → download via CLI tool.

**Published baselines:**

| Method | R@1 IoU=0.3 | R@1 IoU=0.5 | R@5 IoU=0.3 | R@5 IoU=0.5 |
|--------|-------------|-------------|-------------|-------------|
| VSLNet (supervised) | 5.45% | 3.12% | 10.74% | 6.63% |
| 2D-TAN (supervised) | 4.52% | 2.02% | — | — |

Note: Very low numbers — finding a 5-second moment in a 20-minute egocentric video is very hard.

---

### 4. Charades-STA

**What it is:** Indoor home activity videos with temporal sentence grounding annotations. Built on top of the Charades dataset.

**Query style:** Free-form natural language descriptions.
- "A person opens a door"
- "Someone throws a pillow on the sofa"

**GT windows:** Yes — [start, end] in seconds.

**Video length:** Average ~30 seconds. **Too short for Kairos** — even shorter than QVHighlights.

**Verdict:** Not suitable. Videos are too short for Kairos's long-video design, and the domain is limited to indoor home activities.

---

### 5. ActivityNet Captions

**What it is:** Dense temporal captioning annotations on top of ActivityNet's 20K YouTube videos. Each video has multiple temporally-grounded captions forming a narrative.

**Query style:** Dense captions (descriptive sentences, not questions).
- "The man picks up a barbell and begins to do curls"
- "A group of people are standing on a field playing soccer"

**GT windows:** Yes — [start, end] in seconds. Multiple segments per video with dense coverage.

**Video length:** Average ~2 minutes, but with a long tail — some videos are 5-10 minutes. **Medium length**.

**Why it's interesting:** Large scale (72K sentences, 20K videos), diverse activities, similar evaluation protocol. Has been widely used for temporal grounding research.

**Why it's problematic:**
- Videos average only ~2 minutes (similar to QVHighlights)
- YouTube link rot: ~30-40% of videos are now unavailable
- ActivityNet was designed for activity recognition, not moment retrieval — the temporal annotations are dense captions rather than retrieval queries

**Verdict:** Similar to QVHighlights in video length and limitations. No clear advantage over what we already have.

---

### 6. TACoS

**What it is:** 127 cooking videos from MPII Cooking Activities with ~18K temporal grounding annotations.

**Video length:** Average ~4.8 minutes. **Short, narrow domain.**

**Verdict:** Too small (127 videos), too short, and cooking-only domain. Not suitable for Kairos.

---

## The Real Question: Which Datasets Can Kairos Actually Run On?

Kairos requires **raw video files** to run its full pipeline (PySceneDetect → BLIP → YOLO → Whisper → AST → LLM → Embeddings). This eliminates datasets that only distribute pre-extracted features.

| Dataset | Raw video available? | Can run Kairos pipeline? |
|---------|---------------------|-------------------------|
| QVHighlights | ~1% via yt-dlp | Yes (16 videos confirmed) |
| MAD | No (movies not distributed, NDA + self-source) | No (without sourcing movies yourself) |
| Ego4D NLQ | Yes (license agreement + download) | Yes (but egocentric domain mismatch) |
| Charades-STA | Yes (downloadable) | Yes (but videos too short) |
| ActivityNet Captions | ~60-70% via yt-dlp | Yes (but videos still short) |
| TACoS | Yes (MPII Cooking) | Yes (but too small and narrow) |

---

## Recommendation

### Short term: Stay with QVHighlights and maximize current benchmark

**Why:** We already have the infrastructure built, results to compare against, and a clear improvement path.

**Action items:**
1. **Implement scene merging** — already proved it works (R@1 IoU=0.5: 25% → 50%)
2. **Run the full benchmark** with `--max-videos 999` to evaluate all available videos (currently only 16)
3. **Try different merge gap thresholds** (2s, 5s, 10s) to find the optimal setting
4. **Report final zero-shot numbers** vs. Moment-DETR / QD-DETR / UniVTG supervised baselines

This gives us a publishable baseline: "Kairos achieves X% R@1 IoU=0.5 zero-shot on QVHighlights, compared to Y% for supervised Moment-DETR."

### Medium term: Add Ego4D NLQ as a second benchmark

**Why:** It has the closest fit for Kairos among available long-video datasets:
- Videos are 8-20 minutes (4-8x longer than QVHighlights)
- Natural language queries with GT windows — same evaluation protocol
- Same metrics (R@1/5 IoU=0.3/0.5) — directly comparable to our QVHighlights results
- Raw video is downloadable (with license)
- Published baselines exist for comparison

**Concerns:**
- Egocentric video is a domain shift for Kairos (first-person vs. third-person)
- PySceneDetect behavior on egocentric video is untested — may need different detection thresholds
- Large download (~5.4 TB full dataset, but subsets available)

**What to do:** Request Ego4D license, download a small subset (50-100 videos), run the same benchmark infrastructure (our metrics code works for any dataset with `[query, [start, end]]` format), and compare.

### Not recommended right now: MAD

**Why not:** Despite having ideal video lengths (1.85-hour movies), the access barrier is too high:
- NDA process
- Movies must be sourced independently (legally complex)
- Without raw video, Kairos pipeline cannot run
- Only pre-extracted CLIP/I3D features are provided

MAD would be valuable if Kairos ever supports feature-based input (bypassing its own pipeline), but for benchmarking the full end-to-end system, it is not practical.

---

## Summary Decision Matrix

| Priority | Dataset | Video Length | Action | Effort |
|----------|---------|-------------|--------|--------|
| **1 (Now)** | QVHighlights | 2.5 min | Finish benchmark: scene merging + full video run | Low — code exists |
| **2 (Next)** | Ego4D NLQ | 8-20 min | Request license, download subset, adapt loader | Medium |
| **3 (Later)** | MAD | ~2 hrs | Only if raw movies can be sourced | High |
| Skip | Charades-STA | 30 sec | Too short | — |
| Skip | ActivityNet | ~2 min | No advantage over QVHighlights | — |
| Skip | TACoS | ~5 min | Too small, cooking only | — |

---

## Can We Compare Across Datasets?

Yes — all these datasets use the **same evaluation protocol**: R@K at IoU thresholds. If we run Kairos on both QVHighlights and Ego4D NLQ, we can directly compare:

```
Kairos zero-shot performance:
  QVHighlights (2.5-min videos): R@1 IoU=0.5 = 50.0%
  Ego4D NLQ   (8-20 min videos): R@1 IoU=0.5 = ??%

If Ego4D is higher → Kairos benefits from longer videos (more scenes, better context)
If Ego4D is lower  → Egocentric domain is harder, or PySceneDetect struggles with first-person video
```

The absolute numbers are not directly comparable (different videos, different queries, different difficulty), but the **relative ranking** (how far from supervised baselines) tells us where Kairos's architecture shines.

---

## References

1. Lei, J., et al. "QVHighlights: Detecting Moments and Highlights in Videos via Natural Language Queries." NeurIPS 2021.
2. Soldan, M., et al. "MAD: A Scalable Dataset for Language Grounding in Videos from Movie Audio Descriptions." CVPR 2022.
3. Grauman, K., et al. "Ego4D: Around the World in 3,000 Hours of Egocentric Video." CVPR 2022.
4. Gao, J., et al. "TALL: Temporal Activity Localization via Language Query." ICCV 2017.
5. Krishna, R., et al. "Dense-Captioning Events in Videos." ICCV 2017.
6. Regneri, M., et al. "Grounding Action Descriptions in Videos." TACL 2013.
