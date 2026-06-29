# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-23 13:49:29 UTC | 1_HvY7N0XVA_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 167.739 | 0.869 | 40.402 | 21.004 | 39.515 | 27.172 | 2.542 |
| 2026-06-27 15:17:57 UTC | 1_HvY7N0XVA_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 100.240 | 0.852 | 39.698 | 8.436 | 5.477 | 8.118 | 2.554 |

## 2026-06-23 13:49:29 UTC | 1_HvY7N0XVA_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/1_HvY7N0XVA_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `167.739` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.869 |
| save_clips | - |
| sample_frames | 0.613 |
| caption_frames | 25.059 |
| sample_fps | 2.008 |
| detect_object_yolo | 7.116 |
| audio_scan | 8.512 |
| asr_timings | 13.274 |
| ast_timings | 18.608 |
| describe_scenes | 21.004 |
| summarize_scenes | 39.515 |
| synthesize_synopsis | 27.172 |
| make_embedding | 2.542 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 25.678 |
| branch_yolo_total | 9.130 |
| branch_audio_total | 40.402 |

## 2026-06-27 15:17:57 UTC | 1_HvY7N0XVA_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/1_HvY7N0XVA_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `100.240` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.852 |
| save_clips | - |
| sample_frames | 0.616 |
| caption_frames | 24.018 |
| sample_fps | 1.992 |
| detect_object_yolo | 7.058 |
| audio_scan | 8.602 |
| asr_timings | 12.355 |
| ast_timings | 18.733 |
| describe_scenes | 8.436 |
| summarize_scenes | 5.477 |
| synthesize_synopsis | 8.118 |
| make_embedding | 2.554 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 24.640 |
| branch_yolo_total | 9.055 |
| branch_audio_total | 39.698 |
