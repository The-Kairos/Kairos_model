# Strategies to Improve Kairos Moment Retrieval

Concrete approaches to close the gap between Kairos (mAP Avg=20.64) and current training-free SOTA (mAP Avg=40.32).

---

## Strategy 1: Query Decomposition

**Gap addressed:** Kairos treats the query as a monolithic embedding. Newer methods decompose queries.

**What to do:** Before embedding the query, use an LLM to decompose it into sub-events or extract key concepts.

**Example:**
- Input query: "a man walks his dog then sits on a bench"
- Decomposed: ["man walking", "dog being walked", "sitting on bench"]
- Retrieve scenes matching each sub-query, then merge results

**Precedent:** Moment-GPT uses LLaMA-3 for query debiasing. TFVTG decomposes into sub-events with temporal relations. GranAlign generates queries at multiple granularity levels.

**Expected impact:** Medium. Helps with compositional queries.

**Implementation effort:** Low — add an LLM call before the embedding step.

---

## Strategy 2: Two-Stage Retrieval (Retrieve + Rerank)

**Gap addressed:** Kairos does single-pass embedding retrieval. Adding LLM reranking is a proven improvement.

**What to do:**
1. Stage 1: Current embedding retrieval (fast, high recall) — retrieve top-20 scenes
2. Stage 2: LLM reranking — send the query + top-20 scene descriptions to an LLM, ask it to rank by relevance

**Precedent:** "Decoupling Semantics and Logic" (ACL 2026, [arXiv:2606.07924](https://arxiv.org/abs/2606.07924)) uses exactly this pattern — coarse dense retrieval then fine LLM reranking. They also strategically exclude noisy modalities (OCR/ASR) from Stage 1 but include them in Stage 2. (Note: that system won the MAGMaR competition, which is a different task — cross-video corpus search — but the two-stage retrieval strategy applies to within-video retrieval too.)

**Expected impact:** High. Reranking catches semantic nuances that embedding similarity misses.

**Implementation effort:** Medium — add an LLM call after retrieval.

---

## Strategy 3: Finer Temporal Granularity

**Gap addressed:** Kairos scenes are 5-15 seconds. Short ground truth moments (0-10s) get low IoU. Our short-moment mAP is 5.37 vs 23.06 for long moments.

**What to do:** Split each scene into 2-second sub-clips (matching QVHighlights' base unit). Generate embeddings at both scene level and sub-clip level. Return sub-clip boundaries, not scene boundaries.

**Approach options:**
- **Option A:** Sub-scene splitting — divide each scene description into temporal segments using frame timestamps
- **Option B:** Dual-scale retrieval — retrieve at scene level for coarse filtering, then at sub-clip level for precise boundaries
- **Option C:** Boundary refinement — after retrieving a scene, use a VLM to predict the exact start/end of the relevant moment within the scene

**Expected impact:** High for short moments (could lift mAP from 5.37 to 15+). Moderate overall.

**Implementation effort:** Medium-High. Requires changes to the frame sampling and embedding pipeline.

---

## Strategy 4: Direct VLM Scoring

**Gap addressed:** Kairos embeds text descriptions and matches against query text. Information is lost in the description generation step.

**What to do:** For the retrieval step, replace text-to-text embedding matching with direct VLM scoring — show the VLM the actual video frames + query and ask it to score relevance.

**Precedent:** REZE (mAP Avg=40.32) does exactly this: VLM directly scores each clip against the query. No description generation needed for retrieval.

**Expected impact:** High. Bypasses the description bottleneck.

**Implementation effort:** High. Fundamental change to the retrieval architecture. Would also significantly increase per-query compute cost.

**Trade-off:** This would make Kairos more like REZE/VTimeCoT and less like a "Video RAG" system. It would improve MR numbers but lose the architectural distinctiveness.

---

## Strategy 5: Better Embeddings

**Gap addressed:** Kairos uses Gemini text embeddings. The choice of embedding model affects retrieval quality.

**What to do:**
- Try OpenAI `text-embedding-3-large` (already supported in the codebase)
- Try embedding the scene descriptions with a model that was trained for retrieval (e.g., E5, BGE, Jina)
- Try multimodal embeddings (e.g., CLIP, SigLIP) that encode both text and images in the same space

**Expected impact:** Low-Medium. Marginal gains from better embeddings, unlikely to close the gap alone.

**Implementation effort:** Low — the embedding provider is already configurable.

---

## Strategy 6: Scene Merging Optimization

**Gap addressed:** The 5-second merge gap was not tuned for QVHighlights.

**What to do:** Sweep merge_gap values (0, 2, 5, 10, 15 seconds) and top_k values (3, 5, 7, 10) on a validation set.

**Expected impact:** Low. Might gain 1-3% mAP.

**Implementation effort:** Very low — just parameter sweeps on existing code.

**Note:** This can be done using `--skip-pipeline` with cached outputs. Only the query embedding API calls are needed, not full pipeline re-runs.

---

## Strategy 7: Adaptive Scene Boundaries

**Gap addressed:** PySceneDetect uses visual transitions only. Some moments span across visual cuts, and some occur within a single long shot.

**What to do:**
- Use audio cues (speech topic changes, silence gaps) as additional scene boundary signals
- Allow scenes to overlap (a moment could span two scenes)
- Use the LLM to post-process scene boundaries based on semantic coherence

**Expected impact:** Medium. Better scene boundaries = better temporal localization.

**Implementation effort:** High. Requires changes to the scene cutting module.

---

## Priority Ranking

| # | Strategy | Expected Impact | Effort | ROI |
|---|----------|----------------|--------|-----|
| 1 | Scene Merging Optimization | Low | Very Low | **High** |
| 2 | Query Decomposition | Medium | Low | **High** |
| 3 | Two-Stage Retrieval (Rerank) | High | Medium | **High** |
| 4 | Better Embeddings | Low-Medium | Low | **Medium** |
| 5 | Finer Temporal Granularity | High | Medium-High | **Medium** |
| 6 | Direct VLM Scoring | High | High | **Low** (loses architectural identity) |
| 7 | Adaptive Scene Boundaries | Medium | High | **Low** |

### Recommended order:
1. **First:** Sweep merging parameters (free wins)
2. **Second:** Add query decomposition via LLM (low effort, clear precedent)
3. **Third:** Add LLM reranking stage (proven by MAGMaR winner)
4. **Fourth:** Try better embedding models (quick experiment)
5. **Later:** Finer granularity and boundary refinement (bigger engineering effort)

---

## Realistic Expectations

Even with all optimizations, Kairos is unlikely to match GranAlign (38.23 mAP Avg) or REZE (40.32 mAP Avg) on QVHighlights because:

1. Those methods are **purpose-built for MR** — Kairos is a general-purpose pipeline
2. Those methods use **direct VLM scoring** — Kairos goes through a text description bottleneck
3. Those methods operate at **clip level** — Kairos operates at scene level

The better strategy is to **improve enough to be competitive** (target: mAP Avg 25-30) while emphasizing that:
- Kairos is a general-purpose system, not an MR specialist
- The same pipeline handles MR, summarization, QA, and RAG
- The long-video scalability (7 hours) is the genuine contribution
- A long-video benchmark (MAD) would better showcase this
