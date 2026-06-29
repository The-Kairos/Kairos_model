# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 16:49:21 UTC | Rzu3oZyGzKw_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 188.219 | 0.680 | 75.617 | 19.031 | 6.933 | 29.477 | 3.597 |

## 2026-06-25 16:49:21 UTC | Rzu3oZyGzKw_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/Rzu3oZyGzKw_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `188.219` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.680 |
| save_clips | - |
| sample_frames | 1.032 |
| caption_frames | 39.356 |
| sample_fps | 2.056 |
| detect_object_yolo | 9.026 |
| audio_scan | 14.355 |
| asr_timings | 31.634 |
| ast_timings | 29.620 |
| describe_scenes | 19.031 |
| summarize_scenes | 6.933 |
| synthesize_synopsis | 29.477 |
| make_embedding | 3.597 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 40.395 |
| branch_yolo_total | 11.087 |
| branch_audio_total | 75.617 |
