# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 12:19:55 UTC | OkHhVRpCOxA_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 218.645 | 0.809 | 76.942 | 22.566 | 18.947 | 30.306 | 4.459 |

## 2026-06-25 12:19:55 UTC | OkHhVRpCOxA_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/OkHhVRpCOxA_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `218.645` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.809 |
| save_clips | - |
| sample_frames | 1.345 |
| caption_frames | 49.070 |
| sample_fps | 2.483 |
| detect_object_yolo | 10.316 |
| audio_scan | 15.487 |
| asr_timings | 23.349 |
| ast_timings | 38.097 |
| describe_scenes | 22.566 |
| summarize_scenes | 18.947 |
| synthesize_synopsis | 30.306 |
| make_embedding | 4.459 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 50.420 |
| branch_yolo_total | 12.805 |
| branch_audio_total | 76.942 |
