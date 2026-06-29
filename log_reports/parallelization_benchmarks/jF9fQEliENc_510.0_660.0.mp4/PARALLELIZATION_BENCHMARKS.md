# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 10:36:12 UTC | jF9fQEliENc_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 204.909 | 0.640 | 60.958 | 28.706 | 18.580 | 30.254 | 4.236 |

## 2026-06-26 10:36:12 UTC | jF9fQEliENc_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/jF9fQEliENc_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `204.909` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.640 |
| save_clips | - |
| sample_frames | 1.294 |
| caption_frames | 46.462 |
| sample_fps | 2.237 |
| detect_object_yolo | 10.051 |
| audio_scan | 14.202 |
| asr_timings | 11.212 |
| ast_timings | 35.536 |
| describe_scenes | 28.706 |
| summarize_scenes | 18.580 |
| synthesize_synopsis | 30.254 |
| make_embedding | 4.236 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 47.762 |
| branch_yolo_total | 12.294 |
| branch_audio_total | 60.958 |
