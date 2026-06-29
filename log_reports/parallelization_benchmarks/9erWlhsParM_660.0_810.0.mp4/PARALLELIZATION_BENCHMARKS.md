# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 18:20:19 UTC | 9erWlhsParM_660.0_810.0.mp4 | sequential | gemini | gemini-embedding-001 | 131.937 | 0.617 | 42.747 | 9.371 | 5.000 | 33.702 | 2.654 |

## 2026-06-24 18:20:19 UTC | 9erWlhsParM_660.0_810.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/9erWlhsParM_660.0_810.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `131.937` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.617 |
| save_clips | - |
| sample_frames | 0.601 |
| caption_frames | 26.463 |
| sample_fps | 1.874 |
| detect_object_yolo | 7.540 |
| audio_scan | 12.801 |
| asr_timings | 10.975 |
| ast_timings | 18.963 |
| describe_scenes | 9.371 |
| summarize_scenes | 5.000 |
| synthesize_synopsis | 33.702 |
| make_embedding | 2.654 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 27.070 |
| branch_yolo_total | 9.421 |
| branch_audio_total | 42.747 |
