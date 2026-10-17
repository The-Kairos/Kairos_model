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

## 6. Next decisions

- Improve event selection and inspect upstream narrative accuracy while keeping the 650-750-word requirement fixed across systems where supported. Treat Titanic as a development example because it has been used for prompt tuning.
- Pilot next: **Yes Man (2008)** (comedy/romance), **Baby Driver (2017)** (action/crime), **Get Out (2017)** (horror/mystery/thriller), then **The Martian (2015)** (sci-fi/adventure/drama). Confirm test-split membership; check turning points, endings, and upstream omissions. These broaden genre coverage, without covering every genre.
- Record fresh full-movie timings locally and on the VM. Treat pilots used for prompt tuning as development examples; freeze prompts and scoring before evaluating held-out movies.
- Validate Video ReCap without MovieSum-specific fine-tuning. Use matched movies and documented input/metric differences for direct system comparisons.
- Report aggregate ROUGE, then add BERTScore and a smaller blinded human review of factuality, plot coverage, and coherence.
- Save each movie ID, reference, generated summary, model version, prompt, settings, processing time, failure status, and scores. Keep reference summaries out of generation inputs.
