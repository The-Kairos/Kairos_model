# Pilot: does Kairos beat the published zero-shot baselines?

**No, not across all measures.** The four-movie average is numerically higher on ROUGE-1/2 F1 and lower on ROUGE-L F1 than all four zero-shot baselines. Without Titanic, ROUGE-2 is below FLAN-UL2 but above the other three. BERTScore settings are not aligned, so its higher numbers do not establish a win.

Three movies = Baby Driver, Get Out, The Martian; four adds Titanic. References are the verified MovieSum Wikipedia summaries, kept out of generation. All values use a 0–100 scale. There are **12 local values**, not 10: P/R/F1 for each of four metrics. The paper reports only six.

| Measure | Three-movie average | Four-movie average | Strongest published zero-shot value | Difference: three / four |
|---|---:|---:|---:|---:|
| ROUGE-1 F1 | 27.47 | 31.52 | 23.62 | +3.85 / +7.90 |
| ROUGE-2 F1 | 3.73 | 5.48 | 4.29 | -0.56 / +1.19 |
| ROUGE-L F1 | 13.44 | 14.08 | 22.01 | -8.57 / -7.93 |
| BERTScore precision* | 81.07 | 81.29 | 59.24 | +21.83 / +22.05 |
| BERTScore recall* | 80.35 | 80.83 | 49.57 | +30.78 / +31.26 |
| BERTScore F1* | 80.71 | 81.06 | 53.57 | +27.14 / +27.49 |

*BERTScore differences are arithmetic only, not comparable performance margins. Our unrescaled roberta-large layer-17 scores use standard token truncation; the paper’s equivalent settings are unverified. Each strongest value is selected independently across baselines.

## Margins against each zero-shot baseline

Each cell is **three-movie / four-movie difference in score points**. Positive means numerically higher. No statistical-equivalence threshold was set, so a small difference is not labeled “comparable.”

| Measure | FLAN-UL2 | Vicuna | TextRank + Vicuna | Moving-window Vicuna |
|---|---:|---:|---:|---:|
| ROUGE-1 F1 | +3.85 / +7.90 | +11.12 / +15.17 | +10.33 / +14.38 | +7.91 / +11.96 |
| ROUGE-2 F1 | -0.56 / +1.19 | +0.18 / +1.93 | +0.05 / +1.80 | +0.41 / +2.16 |
| ROUGE-L F1 | -8.57 / -7.93 | -2.00 / -1.36 | -2.03 / -1.39 | -5.13 / -4.49 |
| BERTScore precision* | +28.17 / +28.39 | +32.18 / +32.40 | +21.83 / +22.05 | +26.12 / +26.34 |
| BERTScore recall* | +30.78 / +31.26 | +31.86 / +32.34 | +31.30 / +31.78 | +31.65 / +32.13 |
| BERTScore F1* | +29.84 / +30.19 | +33.64 / +33.99 | +27.14 / +27.49 | +29.18 / +29.53 |

Published baselines: [MovieSum Table 3](https://aclanthology.org/2024.findings-acl.239.pdf#page=4).

## Remaining local measures

The paper does not publish ROUGE precision or recall separately. These six local values have no published counterpart to beat:

| Metric | Three: precision / recall | Four: precision / recall |
|---|---:|---:|
| ROUGE-1 | 26.87 / 28.20 | 30.82 / 32.33 |
| ROUGE-2 | 3.64 / 3.84 | 5.35 / 5.63 |
| ROUGE-L | 13.16 / 13.79 | 13.78 / 14.44 |

## Scope

This compares selected video diagnostics with published screenplay test-set averages, not baseline outputs for the same movies. Titanic was used for tuning, the local cuts remain unverified, and scoring preprocessing may differ. Missing plot events remain a quality concern. These results do not establish overall superiority.

Per-movie scores, all P/R/F1 values, reference checks and hashes are in [results.json](results.json). Rebuild this report with `.venv\Scripts\python.exe evaluation\pilot\report_results.py`. Raw local runs, movies and frozen sources stay untracked.
