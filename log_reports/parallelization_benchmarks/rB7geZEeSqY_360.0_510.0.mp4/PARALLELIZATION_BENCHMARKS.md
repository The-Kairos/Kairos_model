# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 18:31:22 UTC | rB7geZEeSqY_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 134.260 | 0.744 | 43.472 | 13.474 | 7.793 | 22.057 | 3.004 |

## 2026-06-26 18:31:22 UTC | rB7geZEeSqY_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/rB7geZEeSqY_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `134.260` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.744 |
| save_clips | - |
| sample_frames | 0.820 |
| caption_frames | 31.837 |
| sample_fps | 2.038 |
| detect_object_yolo | 7.598 |
| audio_scan | 12.883 |
| asr_timings | 6.476 |
| ast_timings | 24.104 |
| describe_scenes | 13.474 |
| summarize_scenes | 7.793 |
| synthesize_synopsis | 22.057 |
| make_embedding | 3.004 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 32.663 |
| branch_yolo_total | 9.642 |
| branch_audio_total | 43.472 |
