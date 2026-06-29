# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 23:55:26 UTC | _YfQuBuAmHs_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 151.213 | 0.656 | 57.331 | 11.762 | 7.306 | 9.066 | 4.189 |

## 2026-06-25 23:55:26 UTC | _YfQuBuAmHs_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/_YfQuBuAmHs_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `151.213` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.656 |
| save_clips | - |
| sample_frames | 1.137 |
| caption_frames | 46.260 |
| sample_fps | 2.209 |
| detect_object_yolo | 9.856 |
| audio_scan | 11.750 |
| asr_timings | 9.634 |
| ast_timings | 35.939 |
| describe_scenes | 11.762 |
| summarize_scenes | 7.306 |
| synthesize_synopsis | 9.066 |
| make_embedding | 4.189 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 47.403 |
| branch_yolo_total | 12.070 |
| branch_audio_total | 57.331 |
