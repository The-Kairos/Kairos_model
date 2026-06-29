# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 03:29:56 UTC | x6QkZM27EVw_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 119.612 | 0.650 | 61.326 | 7.907 | 4.702 | 6.070 | 2.513 |

## 2026-06-27 03:29:56 UTC | x6QkZM27EVw_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/x6QkZM27EVw_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `119.612` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.650 |
| save_clips | - |
| sample_frames | 0.757 |
| caption_frames | 24.756 |
| sample_fps | 1.970 |
| detect_object_yolo | 7.557 |
| audio_scan | 16.234 |
| asr_timings | 27.206 |
| ast_timings | 17.878 |
| describe_scenes | 7.907 |
| summarize_scenes | 4.702 |
| synthesize_synopsis | 6.070 |
| make_embedding | 2.513 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 25.518 |
| branch_yolo_total | 9.532 |
| branch_audio_total | 61.326 |
