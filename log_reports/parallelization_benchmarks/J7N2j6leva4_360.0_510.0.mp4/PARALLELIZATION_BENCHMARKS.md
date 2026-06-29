# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 05:00:43 UTC | J7N2j6leva4_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 102.802 | 0.662 | 39.104 | 8.477 | 8.461 | 16.627 | 2.020 |

## 2026-06-25 05:00:43 UTC | J7N2j6leva4_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/J7N2j6leva4_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `102.802` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.662 |
| save_clips | - |
| sample_frames | 0.432 |
| caption_frames | 17.183 |
| sample_fps | 1.791 |
| detect_object_yolo | 6.645 |
| audio_scan | 16.025 |
| asr_timings | 10.323 |
| ast_timings | 12.746 |
| describe_scenes | 8.477 |
| summarize_scenes | 8.461 |
| synthesize_synopsis | 16.627 |
| make_embedding | 2.020 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 17.621 |
| branch_yolo_total | 8.441 |
| branch_audio_total | 39.104 |
