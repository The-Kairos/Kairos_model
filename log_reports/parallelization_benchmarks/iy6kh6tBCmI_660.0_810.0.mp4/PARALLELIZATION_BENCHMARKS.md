# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 09:55:06 UTC | iy6kh6tBCmI_660.0_810.0.mp4 | sequential | gemini | gemini-embedding-001 | 172.354 | 0.836 | 51.572 | 18.412 | 10.196 | 36.559 | 3.316 |

## 2026-06-26 09:55:06 UTC | iy6kh6tBCmI_660.0_810.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/iy6kh6tBCmI_660.0_810.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `172.354` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.836 |
| save_clips | - |
| sample_frames | 1.272 |
| caption_frames | 37.737 |
| sample_fps | 2.341 |
| detect_object_yolo | 8.665 |
| audio_scan | 15.046 |
| asr_timings | 9.273 |
| ast_timings | 27.244 |
| describe_scenes | 18.412 |
| summarize_scenes | 10.196 |
| synthesize_synopsis | 36.559 |
| make_embedding | 3.316 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 39.015 |
| branch_yolo_total | 11.012 |
| branch_audio_total | 51.572 |
