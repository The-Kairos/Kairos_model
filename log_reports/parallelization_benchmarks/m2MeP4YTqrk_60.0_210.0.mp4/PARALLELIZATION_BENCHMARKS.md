# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 17:53:06 UTC | m2MeP4YTqrk_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 174.306 | 0.812 | 47.370 | 25.269 | 24.234 | 16.015 | 3.617 |

## 2026-06-26 17:53:06 UTC | m2MeP4YTqrk_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/m2MeP4YTqrk_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `174.306` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.812 |
| save_clips | - |
| sample_frames | 1.401 |
| caption_frames | 41.888 |
| sample_fps | 2.394 |
| detect_object_yolo | 9.809 |
| audio_scan | 6.601 |
| asr_timings | 10.426 |
| ast_timings | 30.334 |
| describe_scenes | 25.269 |
| summarize_scenes | 24.234 |
| synthesize_synopsis | 16.015 |
| make_embedding | 3.617 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 43.295 |
| branch_yolo_total | 12.210 |
| branch_audio_total | 47.370 |
