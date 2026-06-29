# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 23:02:06 UTC | ZudB4C8rtQU_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 164.934 | 0.668 | 59.648 | 14.085 | 8.820 | 11.162 | 4.730 |

## 2026-06-25 23:02:06 UTC | ZudB4C8rtQU_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/ZudB4C8rtQU_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `164.934` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.668 |
| save_clips | - |
| sample_frames | 1.329 |
| caption_frames | 50.357 |
| sample_fps | 2.231 |
| detect_object_yolo | 10.470 |
| audio_scan | 9.572 |
| asr_timings | 11.360 |
| ast_timings | 38.708 |
| describe_scenes | 14.085 |
| summarize_scenes | 8.820 |
| synthesize_synopsis | 11.162 |
| make_embedding | 4.730 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 51.692 |
| branch_yolo_total | 12.707 |
| branch_audio_total | 59.648 |
