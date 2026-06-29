# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 22:05:50 UTC | Ys290kErJzE_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 160.539 | 0.645 | 71.069 | 9.121 | 14.090 | 14.773 | 3.700 |

## 2026-06-25 22:05:50 UTC | Ys290kErJzE_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/Ys290kErJzE_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `160.539` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.645 |
| save_clips | - |
| sample_frames | 1.024 |
| caption_frames | 34.259 |
| sample_fps | 2.114 |
| detect_object_yolo | 8.340 |
| audio_scan | 7.567 |
| asr_timings | 36.091 |
| ast_timings | 27.403 |
| describe_scenes | 9.121 |
| summarize_scenes | 14.090 |
| synthesize_synopsis | 14.773 |
| make_embedding | 3.700 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 35.290 |
| branch_yolo_total | 10.460 |
| branch_audio_total | 71.069 |
