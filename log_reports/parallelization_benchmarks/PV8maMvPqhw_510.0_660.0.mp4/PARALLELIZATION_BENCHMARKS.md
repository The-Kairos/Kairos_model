# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 14:03:40 UTC | PV8maMvPqhw_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 269.742 | 0.668 | 68.368 | 33.568 | 64.642 | 24.360 | 5.385 |

## 2026-06-25 14:03:40 UTC | PV8maMvPqhw_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/PV8maMvPqhw_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `269.742` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.668 |
| save_clips | - |
| sample_frames | 2.114 |
| caption_frames | 55.349 |
| sample_fps | 2.547 |
| detect_object_yolo | 11.320 |
| audio_scan | 15.789 |
| asr_timings | 8.949 |
| ast_timings | 43.622 |
| describe_scenes | 33.568 |
| summarize_scenes | 64.642 |
| synthesize_synopsis | 24.360 |
| make_embedding | 5.385 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 57.469 |
| branch_yolo_total | 13.873 |
| branch_audio_total | 68.368 |
