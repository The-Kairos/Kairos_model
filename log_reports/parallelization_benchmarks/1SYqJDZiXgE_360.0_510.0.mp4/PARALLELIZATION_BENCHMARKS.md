# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-23 13:31:40 UTC | 1SYqJDZiXgE_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 220.728 | 0.642 | 46.930 | 28.412 | 44.102 | 38.188 | 3.962 |
| 2026-06-27 15:05:12 UTC | 1SYqJDZiXgE_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 141.485 | 0.625 | 47.995 | 10.021 | 13.210 | 7.848 | 3.942 |

## 2026-06-23 13:31:40 UTC | 1SYqJDZiXgE_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/1SYqJDZiXgE_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `220.728` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.642 |
| save_clips | - |
| sample_frames | 1.531 |
| caption_frames | 44.771 |
| sample_fps | 2.253 |
| detect_object_yolo | 8.557 |
| audio_scan | 6.424 |
| asr_timings | 7.973 |
| ast_timings | 32.525 |
| describe_scenes | 28.412 |
| summarize_scenes | 44.102 |
| synthesize_synopsis | 38.188 |
| make_embedding | 3.962 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 46.307 |
| branch_yolo_total | 10.815 |
| branch_audio_total | 46.930 |

## 2026-06-27 15:05:12 UTC | 1SYqJDZiXgE_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/1SYqJDZiXgE_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `141.485` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.625 |
| save_clips | - |
| sample_frames | 1.527 |
| caption_frames | 43.935 |
| sample_fps | 2.284 |
| detect_object_yolo | 8.705 |
| audio_scan | 6.453 |
| asr_timings | 8.687 |
| ast_timings | 32.846 |
| describe_scenes | 10.021 |
| summarize_scenes | 13.210 |
| synthesize_synopsis | 7.848 |
| make_embedding | 3.942 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 45.468 |
| branch_yolo_total | 10.994 |
| branch_audio_total | 47.995 |
