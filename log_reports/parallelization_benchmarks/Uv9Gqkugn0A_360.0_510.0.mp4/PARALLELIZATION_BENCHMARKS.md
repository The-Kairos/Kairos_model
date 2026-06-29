# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 19:30:40 UTC | Uv9Gqkugn0A_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 96.812 | 0.796 | 37.874 | 7.178 | 6.460 | 15.603 | 2.045 |

## 2026-06-25 19:30:40 UTC | Uv9Gqkugn0A_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/Uv9Gqkugn0A_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `96.812` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.796 |
| save_clips | - |
| sample_frames | 0.363 |
| caption_frames | 16.620 |
| sample_fps | 1.867 |
| detect_object_yolo | 6.617 |
| audio_scan | 14.934 |
| asr_timings | 10.376 |
| ast_timings | 12.555 |
| describe_scenes | 7.178 |
| summarize_scenes | 6.460 |
| synthesize_synopsis | 15.603 |
| make_embedding | 2.045 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 16.989 |
| branch_yolo_total | 8.490 |
| branch_audio_total | 37.874 |
