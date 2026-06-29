# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 01:35:16 UTC | b_gbAILvCQo_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 166.577 | 0.680 | 61.404 | 15.687 | 10.501 | 8.399 | 4.456 |

## 2026-06-26 01:35:16 UTC | b_gbAILvCQo_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/b_gbAILvCQo_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `166.577` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.680 |
| save_clips | - |
| sample_frames | 1.392 |
| caption_frames | 49.611 |
| sample_fps | 2.349 |
| detect_object_yolo | 10.701 |
| audio_scan | 14.019 |
| asr_timings | 8.785 |
| ast_timings | 38.591 |
| describe_scenes | 15.687 |
| summarize_scenes | 10.501 |
| synthesize_synopsis | 8.399 |
| make_embedding | 4.456 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 51.009 |
| branch_yolo_total | 13.056 |
| branch_audio_total | 61.404 |
