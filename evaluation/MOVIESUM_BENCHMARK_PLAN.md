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

## 3. Scope, allocation, and time

Use the **official 200-movie test split**, divided among **six people: 34 movies each for two people and 33 each for four**. Match movie title, year, and version; keep assignments fixed across models and record unavailable movies and processing failures. Wikipedia references may differ from the released movie or screenplay.

Provisional Kairos estimates for **two-hour movies, processed one at a time**:

| Hardware | Per movie | 33-34 movies/person | 100 movies |
|---|---:|---:|---:|
| Local laptop (GTX 1660 Ti) | ~5 hours | ~165-170 hours | ~500 hours (~21 continuous days) |
| VM (NVIDIA L4; Titanic-based extrapolation) | ~2.7 hours | ~89-91 hours | ~270 hours (~11 continuous days) |

These are extrapolations from historical **summed stage timings**, not fresh end-to-end benchmarks: [local clips](../logs/New%20BLIP%20params/) averaged ~2.5 processing hours/video hour; the [VM Titanic record](../logs/_processed/Titanic.1997.mkv_20260312_201859.json) totals ~4.36 hours for a 3.25-hour movie. Cached reruns make start/end timestamps unreliable. Run a fresh full-movie pilot on each setup before fixing allocations; current optimizations and knowledge-graph extraction may change runtime. For 100 movies locally, budget **500-700 processing hours**, excluding setup, downloads, evaluation, and review. Parallel work requires sufficient compute and API capacity; estimate other models separately.

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

## 5. Outputs and next decisions

Save each movie ID, reference summary, generated summary, model version, prompt, settings, processing time, failure status, and evaluation scores. Keep reference summaries out of model inputs.

The [Titanic sanity check](titanic_sanity/RESULTS.md) has revised-summary ROUGE scores above two published Vicuna zero-shot baselines on all three measures, but this is one movie versus test-set averages, not evidence of overall superiority. Its summary is also too long and misses important events.

Next, control summary length, inspect upstream narrative accuracy, and freeze the prompt and scoring settings before evaluating held-out movies. Titanic is a development example once used for prompt tuning. Report aggregate ROUGE results, then add BERTScore and a smaller blinded human review of factuality, plot coverage, and coherence.

Use the paper's zero-shot scores as context; a direct performance claim requires matched movies and documented input/metric differences. Validate Video ReCap without MovieSum-specific fine-tuning before including it as a separate video baseline.
