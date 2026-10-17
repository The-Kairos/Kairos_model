# Titanic: did Kairos outperform the zero-shot models?

**Numerically, yes against two Vicuna baselines on all three ROUGE measures. But this single-movie check does not establish that Kairos is a better system overall.**

| Paper baseline | How Kairos's Titanic scores compare |
|---|---|
| Vicuna | Higher on all three measures; only slightly higher on ROUGE-L |
| TextRank + Vicuna | Higher on all three measures; only slightly higher on ROUGE-L |
| FLAN-UL2 | Higher on ROUGE-1/2; lower on ROUGE-L |
| Moving-window Vicuna | Higher on ROUGE-1/2; lower on ROUGE-L |

ROUGE measures shared words, phrases, and sequences with the human reference. Higher means more overlap, not necessarily a more accurate story. Published scores come from [MovieSum Table 3](https://aclanthology.org/2024.findings-acl.239.pdf#page=4).

**What we can say:** Kairos's Titanic result is encouraging compared with the published zero-shot scores. **What we cannot yet say:** Kairos outperforms those models across movies. Their scores average screenplay-based tests; ours is one video-derived case, with different inputs, lengths, and potentially metric settings.

The summary still misses important events and is **1,407 words versus the reference's 680**. Next, control summary length and compare systems on the same movies.

The authors' exact zero-shot outputs were not found in the checked [official repository](https://github.com/saxenarohit/MovieSum). The paper includes only a fine-tuned LED example in [Appendix F](https://aclanthology.org/2024.findings-acl.239.pdf#page=8).

[Generated summary](generated_summary.txt) · [Reference](reference_summary.txt) · [Detailed scores and settings](metrics.json) · [Generation metadata](generation_metadata.json)
