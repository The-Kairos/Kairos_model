# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 01:50:20 UTC | c1lX1gd5I78_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 133.486 | 0.814 | 49.523 | 11.581 | 12.322 | 8.164 | 3.282 |

## 2026-06-26 01:50:20 UTC | c1lX1gd5I78_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/c1lX1gd5I78_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `133.486` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.814 |
| save_clips | - |
| sample_frames | 0.920 |
| caption_frames | 34.604 |
| sample_fps | 2.222 |
| detect_object_yolo | 8.660 |
| audio_scan | 11.914 |
| asr_timings | 10.138 |
| ast_timings | 27.462 |
| describe_scenes | 11.581 |
| summarize_scenes | 12.322 |
| synthesize_synopsis | 8.164 |
| make_embedding | 3.282 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 35.530 |
| branch_yolo_total | 10.888 |
| branch_audio_total | 49.523 |
