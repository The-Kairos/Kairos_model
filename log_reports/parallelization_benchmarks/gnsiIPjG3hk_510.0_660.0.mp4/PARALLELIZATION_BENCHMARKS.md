# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 05:40:50 UTC | gnsiIPjG3hk_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 253.655 | 0.809 | 104.386 | 29.399 | 24.544 | 13.946 | 6.814 |

## 2026-06-26 05:40:50 UTC | gnsiIPjG3hk_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/gnsiIPjG3hk_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `253.655` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.809 |
| save_clips | - |
| sample_frames | 1.897 |
| caption_frames | 50.248 |
| sample_fps | 2.829 |
| detect_object_yolo | 12.222 |
| audio_scan | 6.902 |
| asr_timings | 35.948 |
| ast_timings | 61.528 |
| describe_scenes | 29.399 |
| summarize_scenes | 24.544 |
| synthesize_synopsis | 13.946 |
| make_embedding | 6.814 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 52.151 |
| branch_yolo_total | 15.056 |
| branch_audio_total | 104.386 |
