# Moment Retrieval State of the Art — 2026

**Last updated:** 2026-08-22

All numbers are on the **QVHighlights test split** unless noted otherwise.

---

## Leaderboard Links

> **Note (2026-08-22):** Papers With Code (`paperswithcode.com`) is fully dead — all URLs 302 redirect to `huggingface.co/papers/trending`. The Codalab evaluation server for QVHighlights has also been shut down. There is **no active public leaderboard** for QVHighlights moment retrieval. SOTA numbers must be pulled directly from published papers.

### Working Links
- **QVHighlights dataset + eval code (canonical):** https://github.com/jayleicn/moment_detr
- **QVHighlights evaluation protocol:** https://github.com/jayleicn/moment_detr/blob/main/standalone_eval/README.md
- **QVHighlights paper:** https://arxiv.org/abs/2107.09609
- **Codalab (historical, shut down):** https://codalab.lisn.upsaclay.fr/competitions/6937
- **Raw video tarball:** https://nlp.cs.unc.edu/data/jielei/qvh/qvhilights_videos.tar.gz

### Other Benchmarks
- **Ego4D Challenge (CVPR):** https://ego4d-data.org/
- **MAD Benchmark:** https://github.com/Soldelli/MAD

---

## Supervised Methods on QVHighlights Test

| Method | Year | Venue | R1@0.5 | R1@0.7 | mAP@0.5 | mAP@0.75 | mAP Avg | Paper |
|--------|------|-------|--------|--------|---------|----------|---------|-------|
| MCN | 2017 | ICCV | 11.41 | 2.72 | 24.94 | 8.22 | 10.67 | [arXiv:1708.01641](https://arxiv.org/abs/1708.01641) |
| CAL | 2019 | arXiv | 25.49 | 11.54 | 23.40 | 7.65 | 9.89 | [arXiv:1907.12763](https://arxiv.org/abs/1907.12763) |
| XML | 2020 | ECCV | 41.83 | 30.35 | 44.63 | 31.73 | 32.14 | [arXiv:2001.09099](https://arxiv.org/abs/2001.09099) |
| Moment-DETR | 2021 | NeurIPS | 52.89 | 33.02 | 54.82 | 29.40 | 30.73 | [arXiv:2107.09609](https://arxiv.org/abs/2107.09609) |
| Moment-DETR w/ PT | 2021 | NeurIPS | 59.78 | 40.33 | 60.51 | 35.36 | 36.14 | [arXiv:2107.09609](https://arxiv.org/abs/2107.09609) |
| UMT (V+A) | 2022 | -- | 56.23 | 41.18 | 53.83 | 37.01 | 36.12 | -- |
| UniVTG | 2023 | ICCV | 58.86 | 40.86 | 57.60 | 35.59 | 35.47 | [arXiv:2307.16715](https://arxiv.org/abs/2307.16715) |
| UniVTG w/ PT | 2023 | ICCV | 65.43 | 50.06 | 64.06 | 45.02 | 43.63 | [arXiv:2307.16715](https://arxiv.org/abs/2307.16715) |
| QD-DETR | 2023 | CVPR | 62.40 | 44.98 | 62.62 | 39.88 | 39.86 | [arXiv:2303.13874](https://arxiv.org/abs/2303.13874) |
| EaTR | 2023 | ICCV | 61.36 | 45.79 | 61.86 | 41.91 | -- | [arXiv:2308.06947](https://arxiv.org/abs/2308.06947) |
| CG-DETR | 2024 | -- | 65.43 | 48.38 | 64.51 | 42.77 | ~42.9 | -- |
| TR-DETR | 2024 | AAAI | 64.66 | 48.96 | 63.98 | 43.73 | 42.62 | [arXiv:2401.02309](https://arxiv.org/abs/2401.02309) |
| TR-DETR (V+A) | 2024 | AAAI | 65.05 | 47.67 | 64.87 | 42.98 | 43.10 | [arXiv:2401.02309](https://arxiv.org/abs/2401.02309) |
| Mr. BLIP / Chrono | 2024 | -- | 74.77 | 60.51 | 68.12 | 53.38 | ~51.4 | [arXiv:2406.18113](https://arxiv.org/abs/2406.18113) |
| SG-DETR w/ PT | 2024 | -- | 74.20 | 60.40 | -- | -- | 58.80 | [arXiv:2410.01615](https://arxiv.org/abs/2410.01615) |
| **UniTime-SP** | **2025** | -- | **77.76** | **63.29** | -- | -- | -- | [arXiv:2506.18883](https://arxiv.org/abs/2506.18883) |

**Supervised SOTA: R1@0.5 = 77.76 (UniTime-SP, 2025), mAP Avg = 58.80 (SG-DETR w/ PT, 2024)**

---

## Zero-Shot / Training-Free Methods on QVHighlights Test

This is the **directly relevant** comparison tier for Kairos.

| Method | Year | Venue | R1@0.5 | R1@0.7 | mAP@0.5 | mAP Avg | Approach | Paper |
|--------|------|-------|--------|--------|---------|---------|----------|-------|
| CLIP | 2021 | -- | 16.88 | 5.19 | 18.11 | 7.67 | Single-frame cosine sim | [arXiv:2103.00020](https://arxiv.org/abs/2103.00020) |
| VideoLLaMA ZS | -- | -- | 17.10 | -- | -- | 6.20 | Direct VLM | -- |
| VideoChatGPT ZS | -- | -- | 21.10 | -- | -- | 9.50 | Direct VLM | -- |
| UniVTG ZS | 2023 | ICCV | 25.16 | 8.95 | 27.42 | 10.87 | Pretrained, zero-shot eval | [arXiv:2307.16715](https://arxiv.org/abs/2307.16715) |
| **Kairos** | **2026** | -- | **38.91** | **22.83** | **36.95** | **20.64** | **Scene pipeline + embedding retrieval** | -- |
| UniTime-Zero | 2025 | -- | 41.03 | -- | -- | -- | Generative MLLM, zero-shot | [arXiv:2506.18883](https://arxiv.org/abs/2506.18883) |
| Moment-GPT | 2025 | AAAI | 58.30 | 37.70 | 55.10 | 35.00 | 3-MLLM pipeline (LLaMA-3 + MiniGPT-v2 + VideoChatGPT) | [arXiv:2501.07972](https://arxiv.org/abs/2501.07972) |
| GranAlign | 2026 | AAAI | 59.92 | 39.30 | 58.94 | 38.23 | Granularity alignment, training-free | [arXiv:2601.00584](https://arxiv.org/abs/2601.00584) |
| REZE | 2026 | arXiv | -- | -- | -- | 40.32 | VLM scoring + deterministic algorithms | [arXiv:2608.04480](https://arxiv.org/abs/2608.04480) |
| SG-DETR ZS | 2024 | -- | 63.90 | 49.60 | 67.50 | 48.30 | Pretrained on InterVid-MR (1M), zero-shot on QVH | [arXiv:2410.01615](https://arxiv.org/abs/2410.01615) |

**Training-free SOTA: R1@0.5 = 59.92 (GranAlign), mAP Avg = 40.32 (REZE)**

### Where Kairos Sits

```
CLIP --------- UniVTG ZS ---- Kairos ------- UniTime-Zero -- Moment-GPT ---- GranAlign ----- REZE
7.67           10.87          20.64          --              35.00           38.23           40.32
                                                                                        (mAP Avg)
```

Kairos is **2.7x above CLIP** and **1.9x above UniVTG ZS**, but **1.7x below Moment-GPT** and **2x below REZE**.

---

## Other Zero-Shot Methods (Different Evaluation Protocols)

These report mIoU rather than mAP, or evaluate on different splits/datasets:

| Method | Year | Venue | Metric | Score | Dataset | Paper |
|--------|------|-------|--------|-------|---------|-------|
| VTimeCoT (GPT-4o) | 2025 | ICCV | R1@0.5 / mIoU | 59.74 / 54.49 | QVHighlights | -- |
| VTimeCoT (Qwen2-VL-7B) | 2025 | ICCV | R1@0.5 / mIoU | 45.79 / 46.21 | QVHighlights | -- |
| TFVTG | 2024 | ECCV | R1@0.5 | 49.97 | Charades-STA | [arXiv:2408.16219](https://arxiv.org/abs/2408.16219) |
| P2S (Point-to-Span) | 2025 | -- | Avg | 14.5% | MAD (hour-long) | [arXiv:2512.10363](https://arxiv.org/abs/2512.10363) |
| DART | 2026 | -- | -- | SOTA | Charades-STA / ActNet | -- |
| TAG | 2025 | -- | -- | SOTA | Charades-STA / ActNet | -- |

---

## Key Papers to Read

### Must-read (directly relevant to Kairos positioning):

1. **Moment-GPT (AAAI 2025)** — [arXiv:2501.07972](https://arxiv.org/abs/2501.07972)
   Training-free 3-stage MLLM pipeline. Most architecturally comparable to Kairos. R1@0.5=58.30.

2. **GranAlign (AAAI 2026)** — [arXiv:2601.00584](https://arxiv.org/abs/2601.00584)
   Training-free granularity alignment. Current zero-shot R1 SOTA. R1@0.5=59.92.

3. **REZE (2026)** — [arXiv:2608.04480](https://arxiv.org/abs/2608.04480)
   VLM clip scoring with deterministic algorithms. Current training-free mAP SOTA. mAP Avg=40.32.

4. **SG-DETR (2024)** — [arXiv:2410.01615](https://arxiv.org/abs/2410.01615)
   Created InterVid-MR (1M pretraining samples). Zero-shot R1@0.5=63.90, mAP Avg=48.30.

5. **UniTime-SP (2025)** — [arXiv:2506.18883](https://arxiv.org/abs/2506.18883)
   Current overall supervised R1 SOTA at 77.76. Also has UniTime-Zero (zero-shot R1@0.5=41.03).

### Should-read (related pipeline systems):

6. **LLoVi (EMNLP 2024)** — [arXiv:2312.17235](https://arxiv.org/abs/2312.17235)
   Caption clips + LLM reasoning. Same paradigm as Kairos, targets QA not MR.

7. **P2S (2025)** — [arXiv:2512.10363](https://arxiv.org/abs/2512.10363)
   First zero-shot framework for hour-long video MR. Relevant if Kairos targets MAD.

8. **TFVTG (ECCV 2024)** — [arXiv:2408.16219](https://arxiv.org/abs/2408.16219)
   LLM query decomposition + VLM scoring. Different approach but same zero-shot tier.
