# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 19:20:15 UTC | UqMooNqP7Hs_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 135.477 | 0.809 | 46.250 | 9.440 | 16.606 | 14.885 | 2.983 |

## 2026-06-25 19:20:15 UTC | UqMooNqP7Hs_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/UqMooNqP7Hs_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `135.477` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.809 |
| save_clips | - |
| sample_frames | 0.821 |
| caption_frames | 31.677 |
| sample_fps | 2.094 |
| detect_object_yolo | 8.498 |
| audio_scan | 12.776 |
| asr_timings | 9.084 |
| ast_timings | 24.381 |
| describe_scenes | 9.440 |
| summarize_scenes | 16.606 |
| synthesize_synopsis | 14.885 |
| make_embedding | 2.983 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 32.505 |
| branch_yolo_total | 10.598 |
| branch_audio_total | 46.250 |
