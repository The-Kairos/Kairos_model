# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 13:06:30 UTC | k5pnAks5jDs_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 235.077 | 0.793 | 63.398 | 24.607 | 49.519 | 24.637 | 4.761 |

## 2026-06-26 13:06:30 UTC | k5pnAks5jDs_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/k5pnAks5jDs_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `235.077` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.793 |
| save_clips | - |
| sample_frames | 1.222 |
| caption_frames | 51.925 |
| sample_fps | 2.414 |
| detect_object_yolo | 10.382 |
| audio_scan | 14.042 |
| asr_timings | 10.483 |
| ast_timings | 38.865 |
| describe_scenes | 24.607 |
| summarize_scenes | 49.519 |
| synthesize_synopsis | 24.637 |
| make_embedding | 4.761 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 53.152 |
| branch_yolo_total | 12.802 |
| branch_audio_total | 63.398 |
