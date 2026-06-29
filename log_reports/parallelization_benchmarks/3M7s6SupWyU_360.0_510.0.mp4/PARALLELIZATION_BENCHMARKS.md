# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-23 16:18:28 UTC | 3M7s6SupWyU_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 133.179 | 0.659 | 41.422 | 14.839 | 9.311 | 24.477 | 2.878 |
| 2026-06-24 10:14:27 UTC | 3M7s6SupWyU_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 116.877 | 0.680 | 41.177 | 10.491 | 9.310 | 13.656 | 2.762 |

## 2026-06-23 16:18:28 UTC | 3M7s6SupWyU_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/3M7s6SupWyU_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `133.179` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.659 |
| save_clips | - |
| sample_frames | 0.669 |
| caption_frames | 28.384 |
| sample_fps | 1.905 |
| detect_object_yolo | 7.267 |
| audio_scan | 9.555 |
| asr_timings | 10.612 |
| ast_timings | 21.246 |
| describe_scenes | 14.839 |
| summarize_scenes | 9.311 |
| synthesize_synopsis | 24.477 |
| make_embedding | 2.878 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 29.059 |
| branch_yolo_total | 9.177 |
| branch_audio_total | 41.422 |

## 2026-06-24 10:14:27 UTC | 3M7s6SupWyU_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/3M7s6SupWyU_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `116.877` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.680 |
| save_clips | - |
| sample_frames | 0.700 |
| caption_frames | 27.345 |
| sample_fps | 1.956 |
| detect_object_yolo | 7.397 |
| audio_scan | 9.589 |
| asr_timings | 10.196 |
| ast_timings | 21.383 |
| describe_scenes | 10.491 |
| summarize_scenes | 9.310 |
| synthesize_synopsis | 13.656 |
| make_embedding | 2.762 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 28.051 |
| branch_yolo_total | 9.359 |
| branch_audio_total | 41.177 |
