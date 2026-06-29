# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-23 12:10:55 UTC | 0kAcEdn-C1M_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 138.428 | 0.798 | 53.731 | 24.546 | 9.757 | 13.023 | 3.034 |
| 2026-06-27 14:04:18 UTC | 0kAcEdn-C1M_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 135.058 | 0.858 | 58.588 | 8.271 | 8.818 | 10.070 | 3.030 |

## 2026-06-23 12:10:55 UTC | 0kAcEdn-C1M_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/0kAcEdn-C1M_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `138.428` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.798 |
| save_clips | - |
| sample_frames | 0.769 |
| caption_frames | 21.560 |
| sample_fps | 2.087 |
| detect_object_yolo | 7.873 |
| audio_scan | 10.741 |
| asr_timings | 18.933 |
| ast_timings | 24.049 |
| describe_scenes | 24.546 |
| summarize_scenes | 9.757 |
| synthesize_synopsis | 13.023 |
| make_embedding | 3.034 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 22.334 |
| branch_yolo_total | 9.966 |
| branch_audio_total | 53.731 |

## 2026-06-27 14:04:18 UTC | 0kAcEdn-C1M_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/0kAcEdn-C1M_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `135.058` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.858 |
| save_clips | - |
| sample_frames | 0.806 |
| caption_frames | 32.831 |
| sample_fps | 2.136 |
| detect_object_yolo | 8.167 |
| audio_scan | 9.713 |
| asr_timings | 24.478 |
| ast_timings | 24.388 |
| describe_scenes | 8.271 |
| summarize_scenes | 8.818 |
| synthesize_synopsis | 10.070 |
| make_embedding | 3.030 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 33.642 |
| branch_yolo_total | 10.308 |
| branch_audio_total | 58.588 |
