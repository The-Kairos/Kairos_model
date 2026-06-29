# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 02:00:19 UTC | v0x-YFvZXZY_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 153.563 | 0.792 | 56.995 | 11.590 | 8.360 | 14.510 | 3.814 |

## 2026-06-27 02:00:19 UTC | v0x-YFvZXZY_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/v0x-YFvZXZY_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `153.563` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.792 |
| save_clips | - |
| sample_frames | 1.353 |
| caption_frames | 43.187 |
| sample_fps | 2.391 |
| detect_object_yolo | 9.173 |
| audio_scan | 12.933 |
| asr_timings | 11.395 |
| ast_timings | 32.658 |
| describe_scenes | 11.590 |
| summarize_scenes | 8.360 |
| synthesize_synopsis | 14.510 |
| make_embedding | 3.814 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 44.547 |
| branch_yolo_total | 11.570 |
| branch_audio_total | 56.995 |
