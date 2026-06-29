# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 23:28:02 UTC | tLy1eqDOrMU_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 1641.047 | 0.793 | 1548.180 | 13.294 | 9.442 | 7.900 | 3.550 |

## 2026-06-26 23:28:02 UTC | tLy1eqDOrMU_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/tLy1eqDOrMU_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `1641.047` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.793 |
| save_clips | - |
| sample_frames | 1.526 |
| caption_frames | 43.030 |
| sample_fps | 2.350 |
| detect_object_yolo | 9.526 |
| audio_scan | 14.297 |
| asr_timings | 1503.573 |
| ast_timings | 30.300 |
| describe_scenes | 13.294 |
| summarize_scenes | 9.442 |
| synthesize_synopsis | 7.900 |
| make_embedding | 3.550 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 44.562 |
| branch_yolo_total | 11.882 |
| branch_audio_total | 1548.180 |
