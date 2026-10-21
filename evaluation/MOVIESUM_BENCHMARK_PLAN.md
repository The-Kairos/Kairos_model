# Kairos MovieSum Benchmark Plan

## 1. Reference dataset: MovieSum

[MovieSum (ACL 2024)](https://aclanthology.org/2024.findings-acl.239/) introduces **2,200 English screenplay–summary pairs**. The summaries are human-written Wikipedia plot summaries; these are the references we want to use.

The paper also analyzes the dataset and tests existing text summarization models: **screenplay → generated summary → comparison with the Wikipedia summary**. Its experiments cover **simple/extractive baselines, zero-shot models, and fine-tuned models**. We will compare with its **zero-shot results only**, since Kairos uses pretrained models without MovieSum-specific fine-tuning. Our main focus remains the human reference summaries for video evaluation.

- [Dataset and splits](https://huggingface.co/datasets/rohitsaxena/MovieSum)
- [MovieSum code](https://github.com/saxenarohit/MovieSum)
- [Full paper](https://aclanthology.org/2024.findings-acl.239.pdf)

## 2. Our experiment

**Full movie → Kairos → written summary → compare with the matching MovieSum reference.**

Run other video summarizers on the same movies with consistent summary instructions and target length. Each system receives one full video file; it may segment or sample the video internally.

Our first candidate is **[Video ReCap: Recursive Captioning of Hour-Long Videos (CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024/papers/Islam_Video_ReCap_Recursive_Captioning_of_Hour-Long_Videos_CVPR_2024_paper.pdf)**:

- [Code and pretrained model instructions](https://github.com/md-mohaiminul/VideoRecap)
- [Full-video demo notebook](https://github.com/md-mohaiminul/VideoRecap/blob/master/demo.ipynb)

Its demo accepts a video file and produces a whole-video summary through internal captioning stages. We still need to validate setup, runtime, and suitability for movies; its original evaluation focuses on egocentric activities.

## 3. Scope and total processing time

Use the **official 200-movie test split**. Match movie title, year, and version; use the same movies across models and record unavailable movies and processing failures. Wikipedia references may differ from the released movie or screenplay.

Provisional Kairos estimates for **two-hour movies, processed one at a time**:

| Hardware | Per movie | 100 movies total | 200 movies total |
|---|---:|---:|---:|
| Local laptop (GTX 1660 Ti) | ~5 hours | ~500 hours (~21 days) | ~1,000 hours (~42 days) |
| VM (NVIDIA L4; Titanic-based extrapolation) | ~2.7 hours | ~270 hours (~11 days) | ~540 hours (~23 days) |

These are extrapolations from historical **summed stage timings**, not fresh end-to-end benchmarks: [local clips](../logs/New%20BLIP%20params/) averaged ~2.5 processing hours/video hour; the [VM Titanic record](../logs/_processed/Titanic.1997.mkv_20260312_201859.json) totals ~4.36 hours for a 3.25-hour movie. Cached reruns make start/end timestamps unreliable. Run a fresh full-movie pilot on each setup before fixing the schedule; current optimizations and knowledge-graph extraction may change runtime. Days assume continuous operation on one machine. Locally, budget **500-700 hours for 100 movies** or **1,000-1,400 hours for all 200**, excluding setup, downloads, retries, evaluation, and review. Parallel work requires sufficient compute and API capacity; estimate other models separately.

## 4. Evaluation measures and frameworks

- **[ROUGE-1/2/L](https://aclanthology.org/W04-1013/):** word and sequence overlap; retain as baselines for comparison with MovieSum.
- **[BERTScore](https://arxiv.org/abs/1904.09675):** semantic similarity using contextual token embeddings; tolerates different wording. Report precision, recall, and F1.
- **[QAEval](https://aclanthology.org/2021.tacl-1.47/):** question-based reference coverage; checks whether the generated summary can answer questions derived from the human summary.
- **[QAPyramid](https://arxiv.org/abs/2412.07096):** fine-grained content coverage through questions about individual reference facts; helps identify omissions.
- **[FABLES](https://arxiv.org/abs/2404.01261):** an evaluation framework and annotated dataset for narrative faithfulness and content selection, rather than a standalone similarity score.
- **[STORYSUMM](https://aclanthology.org/2024.emnlp-main.557/):** a benchmark for testing whether evaluators detect narrative factual errors, rather than a summary similarity metric.
- **[BooookScore](https://arxiv.org/abs/2310.00785):** long-summary coherence evaluation based on sentence-level error detection; complements factuality and coverage checks.
- **[BARTScore](https://arxiv.org/abs/2106.11520):** generation-probability-based scoring, with variants targeting informativeness, fluency, or factuality.
- **[G-Eval](https://aclanthology.org/2023.emnlp-main.153/):** LLM-based evaluation using explicit criteria; useful for relevance, coherence, fluency, and consistency, with human validation.

**Initial choice:** ROUGE and BERTScore baselines, QAEval or QAPyramid for plot coverage, and blinded human review of factuality and coherence. Verify claims against movie evidence where possible; absence from the reference does not establish an error.

## 5. Sanity test results

The [Titanic sanity check](titanic_sanity/RESULTS.md) regenerated final synthesis from the saved narrative, without rerunning video processing. The summary is **714 words**, within the validated 650-750-word range, versus the reference's 680. ROUGE-1/2/L F1 is **43.67 / 10.72 / 15.99** on a 0-100 scale; all three improved after shortening.

Compared numerically with [MovieSum Table 3](https://aclanthology.org/2024.findings-acl.239.pdf#page=4), Kairos is **higher than zero-shot Vicuna and TextRank + Vicuna on all three ROUGE measures**. Against zero-shot FLAN-UL2 and Moving-window Vicuna, it is higher on ROUGE-1/2 but lower on ROUGE-L. This is one video-derived development example versus published screenplay-based test-set averages, so it does not establish overall superiority. Important plot events are still missing.

## 6. Pilot test results

**Does Kairos beat the published zero-shot baselines across all measures? No.** The four-movie average is numerically higher on ROUGE-1/2 F1 and lower on ROUGE-L F1 than all four baselines. The three-movie average is higher on ROUGE-1 than all four, but its ROUGE-2 is below FLAN-UL2. BERTScore settings remain unaligned, so higher raw values do not establish an advantage.

| Measure | Three-movie average | Four-movie average | Difference from strongest zero-shot: three / four |
|---|---:|---:|---:|
| ROUGE-1 F1 | 27.47 | 31.52 | +3.85 / +7.90 |
| ROUGE-2 F1 | 3.73 | 5.48 | -0.56 / +1.19 |
| ROUGE-L F1 | 13.44 | 14.08 | -8.57 / -7.93 |
| BERTScore precision | 81.07 | 81.29 | Not comparable yet |
| BERTScore recall | 80.35 | 80.83 | Not comparable yet |
| BERTScore F1 | 80.71 | 81.06 | Not comparable yet |

Three movies are Baby Driver, Get Out and The Martian; four adds Titanic. Scores use a 0-100 scale. All reference texts match the official MovieSum Wikipedia summaries (Titanic after whitespace normalization), and ROUGE scores were reproduced. The [concise pilot report](pilot/RESULTS.md) includes all 12 local P/R/F1 values and margins against every zero-shot baseline. The paper publishes only six of these measures; separate ROUGE precision/recall comparisons are unavailable.

These are selected video diagnostics versus published screenplay test-set averages. Titanic was used for tuning; the three local movie cuts remain unverified. Yes Man was skipped. Missing plot events remain a concern, and resumed timings are not fresh end-to-end measurements.

## 7. Next decisions

- Investigate upstream plot omissions and character identification before claiming better summary quality; review turning points and endings against actual movie evidence.
- Verify complete movie versions and replace questionable copies. Run Yes Man only after obtaining a complete, decodable input.
- Align BERTScore model, layer, rescaling and truncation with the paper where possible, and verify ROUGE preprocessing. Keep unmatched scores labeled as contextual comparisons.
- Freeze prompts, summary length (650-750 words), retry policy and scoring before evaluating new held-out movies. Treat Titanic and these pilots as development examples.
- Record fresh full-movie timings locally and on the VM using bounded YOLO batches of 4, FP32 and unchanged 4 FPS sampling; exclude cached restart timings from runtime estimates.
- Validate Video ReCap and other baselines on matched movies without MovieSum-specific fine-tuning. Add question-based coverage evaluation and blinded human review of factuality and coherence.
- Preserve movie IDs, references, generated summaries, model/prompt versions, settings, timings, failures and scores. Keep references out of generation inputs.
