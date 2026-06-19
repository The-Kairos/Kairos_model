# Kairos Benchmarking: Detailed Step-by-Step Evaluation Guide

## What Kairos Outputs (What We Are Evaluating)

Kairos produces four distinct output types. Each needs its own evaluation:

| Output | Format | Example Fields |
|--------|--------|----------------|
| **Scene Segmentation** | List of temporal boundaries | `start_seconds`, `end_seconds`, `start_timecode`, `end_timecode` per scene |
| **Scene Descriptions** | Free-form text per scene | `llm_scene_description` (synthesized from BLIP captions + YOLO objects + Whisper speech + AST sounds) |
| **Synopsis** | Structured JSON | `summary`, `video_highlights[{start, end, highlight}]`, `video_timeline[{timestamp, event}]`, `questions[{question, answer}]`, `suggested_clips[{start, end, description}]` |
| **RAG Q&A** | Free-form answer from retrieval | `create_answer(question, top_matches)` returns text; `top_matches` has cosine similarity scores |

---

## Evaluation 1: RAG Question Answering

### What we are measuring
Can Kairos's RAG system (scene descriptions -> embeddings -> retrieval -> Gemini answer generation) correctly answer questions about a video?

### Metrics

| Metric | What It Measures | Formula | Where Published |
|--------|-----------------|---------|-----------------|
| **MCQ Accuracy** | % of multiple-choice questions answered correctly | `correct / total * 100` | Standard across all VQA benchmarks [1-8] |
| **Accuracy by Duration** | How accuracy changes with video length | Accuracy computed per duration bucket (short/medium/long) | Video-MME [1], LongVideoBench [4] |
| **Retrieval Precision@K** | Are the top-K retrieved chunks relevant to the question? | `relevant_chunks_in_top_k / k` | Standard IR metric |

### Benchmarks to Use

**Primary: Video-MME [1]** (900 videos, 2,700 MCQs, 11s-1hr)
- Why: Has audio/subtitle splits -- directly shows Kairos audio branch value
- Format: 4-way MCQ. Published baselines: Gemini 1.5 Pro 75.0%, GPT-4o 71.9%
- Data: https://huggingface.co/datasets/lmms-lab/Video-MME

**Primary: MLVU [2]** (3min-2hr, CVPR 2025)
- Why: Has Sub-Scene Captioning + Video Summary free-form tasks (directly tests Kairos descriptions)
- Format: 6-way MCQ + free-form generation. Scored by M-Avg and G-Avg
- Data: https://huggingface.co/datasets/MLVU/MVLU

**Optional: HourVideo [3]** (500 videos, 20-120min, NeurIPS 2024)
- Why: Hour-long videos, includes summarization. Human 85% vs best model 37.3% -- big gap
- Data: https://huggingface.co/datasets/HourVideo/HourVideo

### Step-by-Step Procedure

```
Step 1: Download benchmark dataset
   - pip install huggingface_hub
   - huggingface-cli download lmms-lab/Video-MME --local-dir ./benchmarks/video_mme
   - Download QA annotations JSON (questions, options A-D, correct answer)

Step 2: Select video subset
   - Start with 50 videos: ~17 short (<2min), ~17 medium (4-15min), ~16 long (30-60min)
   - This gives statistically meaningful results per duration bucket

Step 3: Process each video through Kairos
   - python main.py process --video <path_to_video>
   - This produces: checkpoint.json (scenes, descriptions, synopsis) + rag_embedding.json

Step 4: For each MCQ, query the RAG system
   - Load rag_embedding.json via load_rag_embeddings()
   - Embed the question text via embed_question()
   - Retrieve top-K contexts via get_top_k_similar(k=10)
   - Generate answer via create_answer(question, top_matches)
   - The answer is free-form text from Gemini

Step 5: Map free-form answer to MCQ option
   - Method A (LLM classification): Prompt GPT-4o with the question, 4 options,
     and Kairos's free-form answer. Ask: "Which option (A/B/C/D) does this
     answer correspond to most closely? Reply with just the letter."
   - Method B (embedding cosine): Embed each MCQ option and the answer,
     pick the option with highest cosine similarity

Step 6: Compute accuracy
   - Overall accuracy = correct / total
   - Accuracy per duration bucket (short/medium/long)
   - Accuracy with audio vs. without audio (re-run pipeline with --no-audio flag)

Step 7: Compare against published baselines
   - Video-MME leaderboard: https://video-mme.github.io/home_page.html
   - Report in table format matching the benchmark's standard presentation
```

### Expected Output Table

| Model | Short | Medium | Long | Overall |
|-------|-------|--------|------|---------|
| GPT-4o | X% | X% | X% | 71.9% |
| Gemini 1.5 Pro | X% | X% | X% | 75.0% |
| **Kairos (Full)** | X% | X% | X% | **X%** |
| Kairos (No Audio) | X% | X% | X% | X% |

---

## Evaluation 2: Scene Segmentation (Temporal Accuracy)

### What we are measuring
Does Kairos's PySceneDetect segmentation produce scene boundaries that align with ground-truth temporal annotations?

### Metrics

| Metric | What It Measures | Formula | Where Published |
|--------|-----------------|---------|-----------------|
| **Temporal IoU** | Overlap between predicted and reference time segments | `intersection(pred, ref) / union(pred, ref)` | ActivityNet Captions [29], SODA [*already implemented in Kairos*] |
| **Precision@IoU=0.5** | % of predicted segments that overlap a ground-truth segment with IoU >= 0.5 | `matched_pred / total_pred` | ActivityNet challenge, dense captioning literature |
| **Recall@IoU=0.5** | % of ground-truth segments that are covered by a predicted segment with IoU >= 0.5 | `matched_ref / total_ref` | Same |
| **F1@IoU=0.5** | Harmonic mean of precision and recall at IoU 0.5 | `2 * P * R / (P + R)` | Same |
| **Mean Temporal IoU** | Average IoU across all matched segment pairs | `mean(IoU for matched pairs)` | Standard |

### Benchmarks to Use

**Primary: ActivityNet Captions [29]** (20K videos, 100K temporal segments)
- Why: Largest temporally-annotated video dataset. Each video has 3-4 annotated temporal segments with start/end timestamps.
- Data: https://cs.stanford.edu/people/ranjaykrishna/densevid/ (annotations JSON) + videos from http://activity-net.org
- Ground truth format: `{"timestamps": [[start1, end1], [start2, end2], ...], "sentences": ["caption1", "caption2", ...]}`

**Secondary: YouCook2 [30]** (2K cooking videos, step-by-step temporal segments)
- Why: Procedure videos with precise temporal step boundaries.
- Data: https://huggingface.co/datasets/lmms-lab/YouCook2

### Step-by-Step Procedure

```
Step 1: Download ActivityNet Captions annotations
   - Download val_1.json and val_2.json from the project page
   - Each entry has video_id, timestamps (list of [start, end] pairs), sentences

Step 2: Download corresponding videos
   - Use yt-dlp (already in requirements.txt) to download by YouTube ID
   - yt-dlp -f "bestvideo[height<=720]+bestaudio" -o "%(id)s.%(ext)s" <video_id>

Step 3: Run Kairos scene detection only
   - Run the pipeline or call get_scene_list() directly from src/scene_cutting.py
   - Extract predicted segments: [{start_seconds, end_seconds}, ...]

Step 4: Align predicted scenes with ground-truth segments
   - Use the temporal_iou() function already in src/benchmarks/metrics/soda_metric.py
   - For each ground-truth segment, find the predicted segment with highest IoU
   - For each predicted segment, find the ground-truth segment with highest IoU

Step 5: Compute metrics
   - For IoU thresholds [0.3, 0.5, 0.7]:
     - Count matched pairs (IoU >= threshold)
     - Precision = matched / num_predicted
     - Recall = matched / num_ground_truth
     - F1 = 2 * P * R / (P + R)
   - Mean IoU across all matched pairs
   - Over-segmentation ratio = num_predicted / num_ground_truth

Step 6: Report
   - Table with P, R, F1 at IoU thresholds 0.3, 0.5, 0.7
   - Mean IoU
   - Analysis: Does Kairos over-segment or under-segment?
```

### Note: Kairos already implements temporal IoU

The file `test/benchmarks/metrics/soda_metric.py` already has `temporal_iou()` and the full SODA framework. You can reuse this directly.

---

## Evaluation 3: Scene Description Quality

### What we are measuring
Are Kairos's per-scene descriptions accurate, complete, and factually grounded in the video content?

### Metrics

| Metric | What It Measures | How It Works | Where Published | Credible? |
|--------|-----------------|-------------|-----------------|-----------|
| **SODA F1** | Combined temporal + caption quality | Matches segments by temporal IoU, then scores matched captions with ROUGE-L. Precision/Recall/F1 over matched pairs. | Fujita et al., ECCV 2020 [A] | **Yes** -- published at ECCV, standard for dense video captioning evaluation. *Already implemented in Kairos.* |
| **EMScore** | Video-text alignment without references | Uses CLIP (ViT-B/32) to compute cosine similarity between video frames and caption text at two levels: coarse (global video vs. global text) and fine-grained (frame-token matching). | Shi et al., **CVPR 2022** [15] | **Yes** -- published at CVPR (top-1 CV venue). Higher human correlation than CIDEr, METEOR, SPICE. Open-source: `pip install emscore` / github.com/ShiYaya/emscore |
| **PAC-S** | Caption quality via augmented contrastive learning | Extends CLIP by training on synthetic positive pairs (generated images + captions). Computes cross-modal similarity in this improved embedding space. Works for both images AND videos. | Sarto et al., **CVPR 2023 (Highlight)** [16] | **Yes** -- CVPR Highlight (top 2.5% of submissions). State-of-art human correlation on video captioning. Open-source: `pip install pacscore` / github.com/aimagelab/pacscore |
| **BERTScore F1** | Semantic similarity between predicted and reference text | Contextual token embeddings from BERT, compute precision/recall/F1 via greedy token matching. | Zhang et al., ICLR 2020 [B] | **Yes** -- standard NLG metric, 10K+ citations. *Already implemented in Kairos.* |
| **Video-ChatGPT 5-Dim Score** | Multi-dimensional quality via LLM-as-judge | GPT scores the description on 5 rubric dimensions (0-5 each): Correctness, Detail, Context, Temporal, Consistency. | Maaz et al., **ACL 2024** [20] | **Yes** -- published at ACL (top-1 NLP venue). De facto standard for video LLM evaluation. Used by 50+ papers. |

### What is PAC-S exactly?

PAC-S = **Positive-Augmented Contrastive learning Score**. It addresses a known weakness of CLIPScore: CLIP was trained on noisy web image-text pairs and sometimes gives high scores to unrelated text. PAC-S fixes this by:

1. Taking the pretrained CLIP model
2. Fine-tuning it with *synthetic positive augmentations* -- generating additional matching image-text pairs to create a denser, more calibrated embedding space
3. Computing cross-modal cosine similarity in this improved space

It was published as a **CVPR 2023 Highlight paper** (top 2.5% of all submissions at the #1 computer vision venue). It achieves the highest Kendall tau-b and Kendall tau-c correlations with human judgments on VATEX-EVAL (video captioning) and Flickr8k/Composite (image captioning), outperforming CLIPScore, CIDEr, SPICE, METEOR, and BERTScore.

**Code**: https://github.com/aimagelab/pacscore

### Step-by-Step Procedure

```
Step 1: Install metrics
   pip install emscore   # EMScore (CVPR 2022)
   pip install pacscore  # PAC-S (CVPR 2023)
   # BERTScore already in Kairos: test/benchmarks/metrics/bertscore_metric.py
   # SODA already in Kairos: test/benchmarks/metrics/soda_metric.py

Step 2: Select evaluation videos
   - Use the same Video-MME / MLVU / ActivityNet videos from Evaluations 1-2
   - This way all evaluations are on the same dataset (stronger paper)

Step 3: Run Kairos pipeline, extract scene descriptions
   - From checkpoint.json, extract for each scene:
     {
       "start": scene["start_seconds"],
       "end": scene["end_seconds"],
       "text": scene["llm_scene_description"]
     }

Step 4A: Reference-free metrics (EMScore + PAC-S)
   - For each scene, extract the corresponding video frames
   - EMScore: 
       from emscore import EMScorer
       scorer = EMScorer()
       score = scorer.score(video_frames, caption_text)
     Returns coarse score + fine-grained score (0 to 1 each)
   - PAC-S:
       from pacscore import PACScore
       scorer = PACScore()
       score = scorer.score(video_frames, caption_text)
     Returns score (0 to 1)
   - Average across all scenes per video, then across videos

Step 4B: Reference-based metrics (SODA + BERTScore) -- only on datasets with references
   - On ActivityNet Captions or SceneWalk (already set up in Kairos):
     from metrics.soda_metric import compute_soda
     result = compute_soda(pred_segments, ref_segments)
     # Returns: precision, recall, f1
   - BERTScore on matched pairs (already implemented):
     from metrics.bertscore_metric import compute_bertscore
     result = compute_bertscore(pred_texts, ref_texts)

Step 4C: LLM-as-Judge (Video-ChatGPT framework [20])
   - For each video, send to GPT-4o:
     Prompt: "You are evaluating a video description. Score each dimension 1-5.
     
     Video description: {kairos_scene_description}
     Ground truth (if available): {reference_text}
     
     Score these dimensions:
     1. Correctness of Information (1-5): Are facts accurate?
     2. Detail Orientation (1-5): Are specific details mentioned?
     3. Contextual Understanding (1-5): Does it capture the main theme?
     4. Temporal Understanding (1-5): Is the sequence of events correct?
     5. Consistency (1-5): Is the description internally consistent?
     
     Return as JSON: {correctness: N, detail: N, context: N, temporal: N, consistency: N}"
   - Run on 50-100 videos
   - Also run on the same videos with Gemini 2.5 Pro and GPT-4o direct video descriptions
   - Compare mean scores across all dimensions

Step 5: Competitor comparison (same videos, same metrics)
   - Gemini 2.5 Pro: Upload video to Gemini API, ask "Describe this video scene by scene"
   - GPT-4o: Sample frames at 1fps, send to GPT-4o, ask for scene descriptions
   - Score both competitor outputs with the same EMScore + PAC-S + LLM-as-judge pipeline

Step 6: Report results table
   - Table: Kairos vs. Gemini vs. GPT-4o on each metric
   - Include 95% confidence intervals (bootstrap 1000 resamples)
```

### Expected Output Table

| System | EMScore | PAC-S | SODA F1 | BERTScore F1 | LLM-Judge (avg 5 dims) |
|--------|---------|-------|---------|-------------|----------------------|
| **Kairos (Full)** | X | X | X | X | X |
| Kairos (No Audio) | X | X | X | X | X |
| Gemini 2.5 Pro | X | X | N/A | N/A | X |
| GPT-4o | X | X | N/A | N/A | X |

---

## Evaluation 4: Synopsis Quality

### What we are measuring
Is Kairos's structured synopsis (summary, highlights, timeline, Q&A pairs, characters) accurate and useful?

### Synopsis Components

| Component | Kairos JSON Field | What to Evaluate |
|-----------|-------------------|-----------------|
| Summary | `synopsis.summary` | Factual accuracy, coverage, coherence |
| Highlights | `synopsis.video_highlights[{start, end, highlight}]` | Are highlighted moments important? Are timestamps correct? |
| Timeline | `synopsis.video_timeline[{timestamp, event}]` | Are events in correct order? Are timestamps accurate? |
| Q&A Pairs | `synopsis.questions[{question, answer}]` | Are questions answerable from the video? Are answers correct? |
| Suggested Clips | `synopsis.suggested_clips[{start, end, description}]` | Do clip boundaries match the described content? |

### Metrics per Component

#### 4A: Summary Evaluation

| Metric | What It Measures | Where Published |
|--------|-----------------|-----------------|
| **MLVU Video Summary G-Avg** | Quality of video summary on standardized benchmark | MLVU, CVPR 2025 [2] |
| **EMScore** (on summary vs. full video) | Does the summary capture the video content? | CVPR 2022 [15] |
| **FActScore** | Factual precision: decompose summary into atomic claims, verify each | Min et al., EMNLP 2023 [27] |
| **LLM-Judge** (summary-specific rubric) | Coherence, completeness, conciseness | G-Eval, EMNLP 2023 [21] |

```
Step 1: MLVU Video Summary task
   - Download MLVU dev set from HuggingFace
   - Run Kairos pipeline on MLVU videos
   - Extract synopsis.summary
   - Submit to MLVU evaluation (G-Avg scoring)
   - Compare against published baselines

Step 2: FActScore on summaries
   - For each synopsis.summary:
     a) Decompose into atomic facts using GPT-4o:
        "Break this text into individual verifiable claims. One claim per line."
        Example: "A man in a red shirt walks across a parking lot" ->
          Claim 1: There is a man
          Claim 2: The man is wearing a red shirt
          Claim 3: The man is walking
          Claim 4: The location is a parking lot
     b) Verify each claim against the video:
        Send video frames + claim to GPT-4o:
        "Is this claim supported by the video? Answer SUPPORTED or NOT SUPPORTED."
     c) FActScore = supported_claims / total_claims
   - Report mean FActScore across all videos
```

#### 4B: Highlights Evaluation

| Metric | What It Measures |
|--------|-----------------|
| **Temporal Precision** | Do highlight timestamps point to the described moment? |
| **Importance Rating** | Are highlighted moments actually important (not trivial)? |

```
Step 1: Temporal accuracy of highlights
   - For each highlight in synopsis.video_highlights:
     - Extract frames at the highlight's start/end timestamps
     - Send frames + highlight text to GPT-4o:
       "Does this text accurately describe what happens at these timestamps?
        Score 1-5 for temporal accuracy."
   - Report mean temporal accuracy score

Step 2: Importance scoring via LLM judge
   - Send all highlights + full video summary to GPT-4o:
     "Rate each highlight 1-5 for importance to understanding the video.
      1 = trivial, 5 = essential."
   - Report mean importance score
```

#### 4C: Timeline Evaluation

| Metric | What It Measures |
|--------|-----------------|
| **Temporal Order Accuracy** | Are timeline events in chronologically correct order? |
| **Event Coverage** | Does the timeline capture all major events? |
| **Timestamp Accuracy** | Do timestamps match when events actually occur? |

```
Step 1: Temporal order verification
   - For each video, extract timeline events with timestamps
   - Check: are timestamps monotonically increasing? (automated check)
   - Send timeline + video to GPT-4o:
     "Are these events listed in correct chronological order? 
      For each event, is the timestamp accurate (within 10 seconds)?
      Report per-event accuracy."
   - Report: % of events with correct ordering, % with accurate timestamps

Step 2: Event coverage
   - Send video to GPT-4o: "List the 5-8 most important events in this video with timestamps."
   - Compare GPT-4o's events against Kairos's timeline (text similarity matching)
   - Coverage = matched_events / total_important_events
```

#### 4D: Q&A Pairs Evaluation

| Metric | What It Measures | Where Published |
|--------|-----------------|-----------------|
| **Answerability** | Can the generated questions be answered from the video? | Standard QA evaluation |
| **Answer Correctness** | Are the generated answers factually correct? | Video-ChatGPT [20] |
| **Question Diversity** | Do questions cover different aspects of the video? | Standard |

```
Step 1: Answer correctness
   - For each Q&A pair in synopsis.questions:
     - Send question + video frames to GPT-4o:
       "Watch this video. Answer this question: {question}. 
        Then compare your answer with: {kairos_answer}. 
        Score the provided answer 1-5 for correctness."
   - Report mean correctness score

Step 2: Cross-validation with RAG
   - Feed each generated question back into Kairos's RAG system
   - Compare RAG answer vs. synopsis answer
   - Agreement rate = % where both give the same answer
   - This validates internal consistency

Step 3: Answerability check
   - Send questions to GPT-4o with video:
     "Can this question be answered from the video? Yes/No."
   - Report % answerable (should be high)
```

---

## Evaluation 5: Ablation Study

### What we are measuring
How much does each Kairos component (BLIP, YOLO, Whisper, AST) contribute?

### Step-by-Step Procedure

```
Step 1: Define ablation configurations
   Full Kairos:    BLIP=on, YOLO=on, Whisper=on, AST=on
   No Audio:       BLIP=on, YOLO=on, Whisper=off, AST=off
   No Objects:     BLIP=on, YOLO=off, Whisper=on, AST=on
   No Captions:    BLIP=off, YOLO=on, Whisper=on, AST=on
   Visual Only:    BLIP=on, YOLO=on, Whisper=off, AST=off

Step 2: Run each configuration on the SAME video subset (50 videos from Video-MME)
   - Modify main.py to skip specific pipeline steps
   - The --redo flag supports re-running specific steps

Step 3: For each configuration, measure:
   - MCQ accuracy on Video-MME (from Evaluation 1)
   - EMScore + PAC-S on scene descriptions (from Evaluation 3)
   - LLM-Judge scores on scene descriptions (from Evaluation 3)
   - FActScore on synopsis summaries (from Evaluation 4)

Step 4: Report delta from full Kairos
   - Example: "Removing the audio branch reduces MCQ accuracy by X%"
   - This proves each component contributes
```

### Expected Ablation Table

| Config | MCQ Acc | EMScore | PAC-S | LLM-Judge (avg) | Delta from Full |
|--------|---------|---------|-------|-----------------|-----------------|
| **Full Kairos** | X% | X | X | X | baseline |
| No Audio | X% | X | X | X | -Y% |
| No Objects | X% | X | X | X | -Y% |
| No Captions | X% | X | X | X | -Y% |
| Visual Only | X% | X | X | X | -Y% |

---

## Complete Metric Summary

| # | Metric | Type | Reference-Free? | Already in Kairos? | Publication |
|---|--------|------|-----------------|-------------------|-------------|
| 1 | MCQ Accuracy | QA | N/A (task-based) | No | All VQA benchmarks |
| 2 | Temporal IoU | Segmentation | Yes | **Yes** (`soda_metric.py`) | ActivityNet [29] |
| 3 | F1@IoU=0.5 | Segmentation | N/A | Partially (IoU exists) | ActivityNet challenge |
| 4 | SODA F1 | Dense captioning | No (needs references) | **Yes** (`soda_metric.py`) | ECCV 2020 |
| 5 | BERTScore F1 | Text similarity | No (needs references) | **Yes** (`bertscore_metric.py`) | ICLR 2020 |
| 6 | ROUGE-L F1 | Text similarity | No (needs references) | **Yes** (`rouge_metric.py`) | ACL 2004 |
| 7 | EMScore | Video-text alignment | **Yes** | No (install needed) | **CVPR 2022** [15] |
| 8 | PAC-S | Video-text alignment | **Yes** | No (install needed) | **CVPR 2023 Highlight** [16] |
| 9 | Video-ChatGPT 5-Dim | LLM-as-judge | Yes (no reference needed) | No (implement) | **ACL 2024** [20] |
| 10 | FActScore | Factual precision | Yes | No (implement) | **EMNLP 2023** [27] |
| 11 | MLVU G-Avg | Generation quality | No (benchmark scoring) | No | **CVPR 2025** [2] |

---

## Priority Order (What To Do First)

1. **Week 1-2**: Set up Video-MME QA pipeline (Evaluation 1). This is the highest-impact result -- direct leaderboard comparison.
2. **Week 2-3**: Install EMScore + PAC-S, run on same videos (Evaluation 3). Quick win, pip-installable.
3. **Week 3-4**: Run ablation study (Evaluation 5). Reuses infrastructure from weeks 1-3.
4. **Week 4-5**: Implement LLM-as-judge + FActScore for synopsis (Evaluations 3C, 4A). API costs ~$50-100.
5. **Week 5-6**: Temporal segmentation evaluation on ActivityNet Captions (Evaluation 2). Independent from above.
6. **Week 6-7**: Competitor comparison (Gemini, GPT-4o) on same benchmarks.

---

## References

[1] Fu et al. "Video-MME." arXiv:2405.21075. https://video-mme.github.io
[2] Zhou et al. "MLVU." CVPR 2025. arXiv:2406.04264. https://github.com/JUNJIE99/MLVU
[3] Chandrasegaran et al. "HourVideo." NeurIPS 2024. arXiv:2411.04998. https://hourvideo.stanford.edu
[4] Wu et al. "LongVideoBench." NeurIPS 2024. arXiv:2407.15754. https://longvideobench.github.io
[15] Shi et al. "EMScore." CVPR 2022. arXiv:2111.08919. https://github.com/ShiYaya/emscore
[16] Sarto et al. "PAC-S." CVPR 2023 Highlight. arXiv:2303.12112. https://github.com/aimagelab/pacscore
[17] Wang et al. "ViCLIP / InternVid." arXiv:2307.06942. https://huggingface.co/OpenGVLab/ViCLIP
[20] Maaz et al. "Video-ChatGPT." ACL 2024. arXiv:2306.05424. https://github.com/mbzuai-oryx/Video-ChatGPT
[21] Liu et al. "G-Eval." EMNLP 2023. arXiv:2303.16634
[27] Min et al. "FActScore." EMNLP 2023. arXiv:2305.14251. https://github.com/shmsw25/FActScore
[29] Krishna et al. "ActivityNet Captions." ICCV 2017. arXiv:1705.00754
[30] Zhou et al. "YouCook2." AAAI 2018. arXiv:1703.09788

[A] Fujita et al. "SODA: Story Oriented Dense Video Captioning Evaluation." ECCV 2020.
[B] Zhang et al. "BERTScore: Evaluating Text Generation with BERT." ICLR 2020. arXiv:1904.09675
