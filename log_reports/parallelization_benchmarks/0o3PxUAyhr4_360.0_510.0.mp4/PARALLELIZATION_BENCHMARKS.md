# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-23 12:22:27 UTC | 0o3PxUAyhr4_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 283.905 | 0.815 | 78.904 | 40.506 | 35.344 | 30.222 | 6.533 |
| 2026-06-27 14:13:39 UTC | 0o3PxUAyhr4_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 219.713 | 0.798 | 80.701 | 18.886 | 9.957 | 8.246 | 6.588 |

## 2026-06-23 12:22:27 UTC | 0o3PxUAyhr4_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/0o3PxUAyhr4_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `283.905` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.815 |
| save_clips | - |
| sample_frames | 1.761 |
| caption_frames | 72.878 |
| sample_fps | 2.736 |
| detect_object_yolo | 12.825 |
| audio_scan | 15.751 |
| asr_timings | 9.977 |
| ast_timings | 53.168 |
| describe_scenes | 40.506 |
| summarize_scenes | 35.344 |
| synthesize_synopsis | 30.222 |
| make_embedding | 6.533 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 74.645 |
| branch_yolo_total | 15.567 |
| branch_audio_total | 78.904 |

## 2026-06-27 14:13:39 UTC | 0o3PxUAyhr4_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/0o3PxUAyhr4_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `219.713` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.798 |
| save_clips | - |
| sample_frames | 1.822 |
| caption_frames | 75.236 |
| sample_fps | 2.776 |
| detect_object_yolo | 13.252 |
| audio_scan | 16.276 |
| asr_timings | 10.678 |
| ast_timings | 53.737 |
| describe_scenes | 18.886 |
| summarize_scenes | 9.957 |
| synthesize_synopsis | 8.246 |
| make_embedding | 6.588 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 77.064 |
| branch_yolo_total | 16.035 |
| branch_audio_total | 80.701 |
