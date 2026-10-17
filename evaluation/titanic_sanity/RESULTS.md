# Titanic: length-controlled sanity check

**Does Kairos beat the published zero-shot baselines? Partly: its Titanic scores are higher than Vicuna and TextRank + Vicuna on all three ROUGE measures. Against FLAN-UL2 and Moving-window Vicuna, it is higher on ROUGE-1/2 but lower on ROUGE-L. It does not beat every zero-shot baseline on every measure.**

| System | ROUGE-1 F1 | ROUGE-2 F1 | ROUGE-L F1 | Kairos comparison |
|---|---:|---:|---:|---|
| **Kairos: Titanic, 714 words** | **43.67** | **10.72** | **15.99** | Current result |
| Vicuna 13B 16K (zero-shot) | 16.35 | 3.55 | 15.44 | Kairos higher on all three |
| TextRank + Vicuna 13B 16K (zero-shot) | 17.14 | 3.68 | 15.47 | Kairos higher on all three |
| FLAN-UL2 8K (zero-shot) | 23.62 | 4.29 | 22.01 | Kairos higher on R-1/2, lower on R-L |
| Moving-window Vicuna 13B 16K (zero-shot) | 19.56 | 3.32 | 18.57 | Kairos higher on R-1/2, lower on R-L |

Scores above use a 0-100 scale; Kairos's saved metrics use 0-1. Baseline values are from [MovieSum Table 3](https://aclanthology.org/2024.findings-acl.239.pdf#page=4).

**This is a numerical comparison, not proof that Kairos is the better system:** Kairos has one video-derived development result; the paper reports screenplay-based test-set averages. We have no matched Titanic outputs from these baselines, and metric settings may differ. A system-level claim requires evaluation on the same held-out movies with matched scoring.

## Effect of controlling summary length

The regenerated summary passes the **650-750-word requirement: 714 words**, versus the reference's 680 (1.05x). Generation used only the saved final narrative, without the reference. This reruns final synthesis and scoring, not the video pipeline.

| Measure | Previous (1,407 words) | Current (714 words) |
|---|---:|---:|
| ROUGE-1 F1 | 0.3805 | 0.4367 |
| ROUGE-2 F1 | 0.0922 | 0.1072 |
| ROUGE-L F1 | 0.1563 | 0.1599 |
| chrF (beta=2) | 0.4376 | 0.4479 |

All overlap scores increased, although ROUGE recall decreased as the summary shortened. Higher overlap does not establish factual accuracy or improved plot coverage, and this development example cannot establish superiority over other systems.

## Remaining coverage problems

The summary still devotes most of its space to opening visuals and wreck exploration. It omits Jack saving Rose from suicide and their developing romance, even though these are in the supplied narrative. It also omits the final reunion described upstream. The opening caption and unusual underwater details are inherited from that narrative and remain unverified against the film.

The portrait scene, framing Jack for theft, Rose returning from the lifeboat, and returning the necklace to the sea are absent from the saved narrative as concrete events; final synthesis cannot safely reconstruct them from prior film knowledge. Next, improve event selection from the available narrative and inspect upstream scene evidence for missing plot events, while keeping the same length requirement and the reference out of generation.

## Reproducibility

The script validates whitespace-separated word count and allows up to five attempts. The successful run took two attempts (908, then 714 words); selection used length only, before scoring. Existing results are archived before explicit regeneration. ROUGE uses stemming and whole-summary LCS; chrF uses the recorded NLTK settings. See the saved metadata for prompts, model, usage, and attempts.

[Generated summary](generated_summary.txt) | [Reference](reference_summary.txt) | [Scores](metrics.json) | [Generation metadata](generation_metadata.json) | [Previous artifacts](archive/20261017T215220890601Z/)
