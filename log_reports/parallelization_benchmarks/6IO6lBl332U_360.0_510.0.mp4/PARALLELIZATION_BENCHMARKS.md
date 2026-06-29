# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-21 10:57:36 UTC | 6IO6lBl332U_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 2088.885 | 1.627 | 2007.434 | 9.091 | 6.200 | 6.238 | 3.693 |
| 2026-06-22 07:26:35 UTC | 6IO6lBl332U_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 0.059 | - | - | - | - | - | - |

## 2026-06-21 10:57:36 UTC | 6IO6lBl332U_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/6IO6lBl332U_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `2088.885` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.627 |
| save_clips | - |
| sample_frames | 2.356 |
| caption_frames | 36.495 |
| sample_fps | 5.971 |
| detect_object_yolo | 8.482 |
| audio_scan | 7.572 |
| asr_timings | 1970.328 |
| ast_timings | 29.525 |
| describe_scenes | 9.091 |
| summarize_scenes | 6.200 |
| synthesize_synopsis | 6.238 |
| make_embedding | 3.693 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 38.856 |
| branch_yolo_total | 14.459 |
| branch_audio_total | 2007.434 |

## 2026-06-22 07:26:35 UTC | 6IO6lBl332U_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/6IO6lBl332U_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `0.059` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | - |
| save_clips | - |
| sample_frames | - |
| caption_frames | - |
| sample_fps | - |
| detect_object_yolo | - |
| audio_scan | - |
| asr_timings | - |
| ast_timings | - |
| describe_scenes | - |
| summarize_scenes | - |
| synthesize_synopsis | - |
| make_embedding | - |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | - |
| branch_yolo_total | - |
| branch_audio_total | - |
