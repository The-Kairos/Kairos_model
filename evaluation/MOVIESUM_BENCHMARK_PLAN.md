# Kairos MovieSum Benchmark Plan

## 1. Reference dataset: MovieSum

[MovieSum (ACL 2024)](https://aclanthology.org/2024.findings-acl.239/) introduces **2,200 English screenplay–summary pairs**. The summaries are human-written Wikipedia plot summaries; these are the references we want to use.

The paper also analyzes the dataset and tests existing text summarization models: **screenplay → generated summary → comparison with the Wikipedia summary**. We can explore those experiments later; our focus is using the reference summaries to evaluate video summarization.

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

## 3. Scope and work allocation

Use the **official 200-movie test split**, rather than all 2,200 entries. Obtain movies legally and match title, year, and version. Wikipedia summaries are human references, but may differ from the released movie or screenplay.

Divide the test set among **six people**:

| Assignment | Movies per person | Total |
|---|---:|---:|
| People 1–2 | 34 | 68 |
| People 3–6 | 33 | 132 |
| **Total** | | **200** |

Keep assignments fixed across models so every system is evaluated on the same movies. Record unavailable movies and processing failures.

## 4. Estimated time per person: Kairos only

First run a small pilot to measure processing time and check summary quality. For sequential processing on one machine per person:

**Time per person ≈ 33–34 movies × average processing hours per movie.**

| Assumed processing time per movie | Time per person |
|---|---:|
| 1 hour | 33–34 hours |
| 2 hours | 66–68 hours |
| 4 hours | 132–136 hours |

These are illustrative compute estimates, not measured Kairos runtimes. Add setup, downloads, retries, and review separately. Six people shorten elapsed time only if compute and API capacity allow simultaneous runs. Estimate other models separately.

## 5. Outputs and next decisions

Save each movie ID, reference summary, generated summary, model version, prompt, settings, processing time, and failure status. Keep reference summaries out of model inputs.

After the pilot, freeze the protocol and choose automatic metrics plus a smaller blinded human review of factual accuracy, plot coverage, and coherence.
