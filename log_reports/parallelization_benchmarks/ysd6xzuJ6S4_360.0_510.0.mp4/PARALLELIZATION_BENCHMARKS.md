# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 05:27:14 UTC | ysd6xzuJ6S4_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 144.699 | 0.758 | 56.579 | 12.105 | 8.461 | 8.077 | 3.570 |

## 2026-06-27 05:27:14 UTC | ysd6xzuJ6S4_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/ysd6xzuJ6S4_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `144.699` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.758 |
| save_clips | - |
| sample_frames | 1.467 |
| caption_frames | 40.917 |
| sample_fps | 2.298 |
| detect_object_yolo | 9.075 |
| audio_scan | 9.643 |
| asr_timings | 16.658 |
| ast_timings | 30.270 |
| describe_scenes | 12.105 |
| summarize_scenes | 8.461 |
| synthesize_synopsis | 8.077 |
| make_embedding | 3.570 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 42.390 |
| branch_yolo_total | 11.379 |
| branch_audio_total | 56.579 |
