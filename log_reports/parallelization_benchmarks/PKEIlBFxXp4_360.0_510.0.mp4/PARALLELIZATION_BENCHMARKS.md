# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 13:22:09 UTC | PKEIlBFxXp4_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 179.307 | 0.632 | 57.335 | 28.699 | 11.442 | 23.711 | 3.645 |

## 2026-06-25 13:22:09 UTC | PKEIlBFxXp4_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/PKEIlBFxXp4_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `179.307` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.632 |
| save_clips | - |
| sample_frames | 0.988 |
| caption_frames | 40.391 |
| sample_fps | 2.060 |
| detect_object_yolo | 8.957 |
| audio_scan | 15.595 |
| asr_timings | 10.807 |
| ast_timings | 30.925 |
| describe_scenes | 28.699 |
| summarize_scenes | 11.442 |
| synthesize_synopsis | 23.711 |
| make_embedding | 3.645 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 41.384 |
| branch_yolo_total | 11.023 |
| branch_audio_total | 57.335 |
