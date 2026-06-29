# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 00:25:39 UTC | FlONE32ZwmQ_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 119.093 | 0.623 | 38.933 | 7.577 | 26.655 | 9.402 | 2.287 |

## 2026-06-25 00:25:39 UTC | FlONE32ZwmQ_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/FlONE32ZwmQ_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `119.093` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.623 |
| save_clips | - |
| sample_frames | 0.491 |
| caption_frames | 22.881 |
| sample_fps | 1.780 |
| detect_object_yolo | 7.069 |
| audio_scan | 9.708 |
| asr_timings | 13.159 |
| ast_timings | 16.058 |
| describe_scenes | 7.577 |
| summarize_scenes | 26.655 |
| synthesize_synopsis | 9.402 |
| make_embedding | 2.287 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 23.378 |
| branch_yolo_total | 8.854 |
| branch_audio_total | 38.933 |
