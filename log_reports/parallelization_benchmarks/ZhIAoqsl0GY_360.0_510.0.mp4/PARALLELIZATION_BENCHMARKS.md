# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 22:41:57 UTC | ZhIAoqsl0GY_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 163.265 | 0.675 | 59.487 | 13.454 | 10.122 | 10.192 | 4.427 |

## 2026-06-25 22:41:57 UTC | ZhIAoqsl0GY_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/ZhIAoqsl0GY_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `163.265` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.675 |
| save_clips | - |
| sample_frames | 1.441 |
| caption_frames | 49.120 |
| sample_fps | 2.327 |
| detect_object_yolo | 10.537 |
| audio_scan | 10.756 |
| asr_timings | 10.416 |
| ast_timings | 38.306 |
| describe_scenes | 13.454 |
| summarize_scenes | 10.122 |
| synthesize_synopsis | 10.192 |
| make_embedding | 4.427 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 50.567 |
| branch_yolo_total | 12.870 |
| branch_audio_total | 59.487 |
