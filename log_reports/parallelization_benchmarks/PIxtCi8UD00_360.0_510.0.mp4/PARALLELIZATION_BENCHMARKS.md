# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 13:12:28 UTC | PIxtCi8UD00_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 205.580 | 0.841 | 61.198 | 25.159 | 34.323 | 22.126 | 3.864 |

## 2026-06-25 13:12:28 UTC | PIxtCi8UD00_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/PIxtCi8UD00_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `205.580` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.841 |
| save_clips | - |
| sample_frames | 1.248 |
| caption_frames | 43.421 |
| sample_fps | 2.352 |
| detect_object_yolo | 9.623 |
| audio_scan | 16.598 |
| asr_timings | 11.538 |
| ast_timings | 33.054 |
| describe_scenes | 25.159 |
| summarize_scenes | 34.323 |
| synthesize_synopsis | 22.126 |
| make_embedding | 3.864 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 44.675 |
| branch_yolo_total | 11.982 |
| branch_audio_total | 61.198 |
