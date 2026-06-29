# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-21 21:05:48 UTC | 2I9-kvemtSU_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 140.477 | 0.792 | 56.305 | 10.075 | 6.940 | 7.461 | 3.571 |

## 2026-06-21 21:05:48 UTC | 2I9-kvemtSU_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/2I9-kvemtSU_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `140.477` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.792 |
| save_clips | - |
| sample_frames | 1.280 |
| caption_frames | 41.383 |
| sample_fps | 2.309 |
| detect_object_yolo | 8.958 |
| audio_scan | 14.856 |
| asr_timings | 11.884 |
| ast_timings | 29.557 |
| describe_scenes | 10.075 |
| summarize_scenes | 6.940 |
| synthesize_synopsis | 7.461 |
| make_embedding | 3.571 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 42.669 |
| branch_yolo_total | 11.272 |
| branch_audio_total | 56.305 |
