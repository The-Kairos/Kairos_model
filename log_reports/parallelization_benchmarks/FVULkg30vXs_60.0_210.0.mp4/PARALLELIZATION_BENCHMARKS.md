# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 00:21:23 UTC | FVULkg30vXs_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 205.477 | 0.785 | 72.344 | 17.634 | 11.126 | 9.172 | 6.498 |

## 2026-06-25 00:21:23 UTC | FVULkg30vXs_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/FVULkg30vXs_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `205.477` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.785 |
| save_clips | - |
| sample_frames | 1.601 |
| caption_frames | 70.738 |
| sample_fps | 2.592 |
| detect_object_yolo | 12.240 |
| audio_scan | 6.510 |
| asr_timings | 12.161 |
| ast_timings | 52.995 |
| describe_scenes | 17.634 |
| summarize_scenes | 11.126 |
| synthesize_synopsis | 9.172 |
| make_embedding | 6.498 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 72.344 |
| branch_yolo_total | 14.837 |
| branch_audio_total | 71.674 |
