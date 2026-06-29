# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 13:54:53 UTC | 7MG8MlfEL-k_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 152.809 | 0.784 | 42.588 | 17.269 | 16.040 | 34.660 | 2.552 |

## 2026-06-24 13:54:53 UTC | 7MG8MlfEL-k_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/7MG8MlfEL-k_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `152.809` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.784 |
| save_clips | - |
| sample_frames | 0.701 |
| caption_frames | 27.306 |
| sample_fps | 2.071 |
| detect_object_yolo | 7.391 |
| audio_scan | 14.969 |
| asr_timings | 8.986 |
| ast_timings | 18.624 |
| describe_scenes | 17.269 |
| summarize_scenes | 16.040 |
| synthesize_synopsis | 34.660 |
| make_embedding | 2.552 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 28.013 |
| branch_yolo_total | 9.468 |
| branch_audio_total | 42.588 |
