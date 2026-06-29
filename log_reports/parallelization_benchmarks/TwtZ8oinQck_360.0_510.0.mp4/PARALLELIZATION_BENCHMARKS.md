# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 18:13:55 UTC | TwtZ8oinQck_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 202.832 | 0.861 | 66.065 | 14.800 | 15.111 | 23.773 | 5.429 |

## 2026-06-25 18:13:55 UTC | TwtZ8oinQck_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/TwtZ8oinQck_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `202.832` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.861 |
| save_clips | - |
| sample_frames | 1.546 |
| caption_frames | 59.757 |
| sample_fps | 2.560 |
| detect_object_yolo | 11.455 |
| audio_scan | 11.946 |
| asr_timings | 9.049 |
| ast_timings | 45.061 |
| describe_scenes | 14.800 |
| summarize_scenes | 15.111 |
| synthesize_synopsis | 23.773 |
| make_embedding | 5.429 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 61.309 |
| branch_yolo_total | 14.020 |
| branch_audio_total | 66.065 |
