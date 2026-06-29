# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 23:31:00 UTC | tLy1eqDOrMU_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 177.003 | 0.812 | 92.579 | 8.148 | 6.366 | 19.314 | 2.979 |

## 2026-06-26 23:31:00 UTC | tLy1eqDOrMU_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/tLy1eqDOrMU_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `177.003` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.812 |
| save_clips | - |
| sample_frames | 1.257 |
| caption_frames | 33.913 |
| sample_fps | 2.262 |
| detect_object_yolo | 7.964 |
| audio_scan | 13.894 |
| asr_timings | 53.599 |
| ast_timings | 25.077 |
| describe_scenes | 8.148 |
| summarize_scenes | 6.366 |
| synthesize_synopsis | 19.314 |
| make_embedding | 2.979 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 35.176 |
| branch_yolo_total | 10.232 |
| branch_audio_total | 92.579 |
