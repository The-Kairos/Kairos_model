# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 13:19:56 UTC | 0H2xFsZtjNY_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 117.155 | 0.811 | 51.369 | 7.649 | 7.208 | 9.213 | 2.550 |

## 2026-06-27 13:19:56 UTC | 0H2xFsZtjNY_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/0H2xFsZtjNY_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `117.155` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.811 |
| save_clips | - |
| sample_frames | 1.009 |
| caption_frames | 26.234 |
| sample_fps | 2.178 |
| detect_object_yolo | 7.537 |
| audio_scan | 15.939 |
| asr_timings | 16.654 |
| ast_timings | 18.767 |
| describe_scenes | 7.649 |
| summarize_scenes | 7.208 |
| synthesize_synopsis | 9.213 |
| make_embedding | 2.550 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 27.249 |
| branch_yolo_total | 9.722 |
| branch_audio_total | 51.369 |
