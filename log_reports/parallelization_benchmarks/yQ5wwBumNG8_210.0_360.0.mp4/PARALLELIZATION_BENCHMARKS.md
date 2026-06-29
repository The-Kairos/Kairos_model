# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 04:59:55 UTC | yQ5wwBumNG8_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 120.730 | 0.649 | 54.404 | 7.567 | 5.455 | 7.687 | 3.025 |

## 2026-06-27 04:59:55 UTC | yQ5wwBumNG8_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/yQ5wwBumNG8_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `120.730` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.649 |
| save_clips | - |
| sample_frames | 0.696 |
| caption_frames | 30.532 |
| sample_fps | 1.966 |
| detect_object_yolo | 7.352 |
| audio_scan | 5.493 |
| asr_timings | 24.765 |
| ast_timings | 24.138 |
| describe_scenes | 7.567 |
| summarize_scenes | 5.455 |
| synthesize_synopsis | 7.687 |
| make_embedding | 3.025 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 31.234 |
| branch_yolo_total | 9.323 |
| branch_audio_total | 54.404 |
