# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 23:52:54 UTC | _YfQuBuAmHs_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 112.538 | 0.684 | 42.591 | 6.060 | 7.519 | 12.378 | 2.746 |

## 2026-06-25 23:52:54 UTC | _YfQuBuAmHs_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/_YfQuBuAmHs_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `112.538` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.684 |
| save_clips | - |
| sample_frames | 0.743 |
| caption_frames | 28.912 |
| sample_fps | 2.008 |
| detect_object_yolo | 7.406 |
| audio_scan | 10.751 |
| asr_timings | 10.459 |
| ast_timings | 21.373 |
| describe_scenes | 6.060 |
| summarize_scenes | 7.519 |
| synthesize_synopsis | 12.378 |
| make_embedding | 2.746 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 29.661 |
| branch_yolo_total | 9.419 |
| branch_audio_total | 42.591 |
