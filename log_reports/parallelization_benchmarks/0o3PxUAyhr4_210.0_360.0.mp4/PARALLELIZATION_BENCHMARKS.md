# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-23 12:17:42 UTC | 0o3PxUAyhr4_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 254.899 | 0.767 | 75.978 | 40.560 | 20.598 | 23.169 | 6.542 |
| 2026-06-27 14:09:58 UTC | 0o3PxUAyhr4_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 215.244 | 0.782 | 77.284 | 19.573 | 9.108 | 8.946 | 6.541 |

## 2026-06-23 12:17:42 UTC | 0o3PxUAyhr4_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/0o3PxUAyhr4_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `254.899` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.767 |
| save_clips | - |
| sample_frames | 1.566 |
| caption_frames | 69.139 |
| sample_fps | 2.573 |
| detect_object_yolo | 12.630 |
| audio_scan | 14.723 |
| asr_timings | 9.848 |
| ast_timings | 51.395 |
| describe_scenes | 40.560 |
| summarize_scenes | 20.598 |
| synthesize_synopsis | 23.169 |
| make_embedding | 6.542 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 70.711 |
| branch_yolo_total | 15.209 |
| branch_audio_total | 75.978 |

## 2026-06-27 14:09:58 UTC | 0o3PxUAyhr4_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/0o3PxUAyhr4_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `215.244` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.782 |
| save_clips | - |
| sample_frames | 1.604 |
| caption_frames | 74.332 |
| sample_fps | 2.652 |
| detect_object_yolo | 12.977 |
| audio_scan | 15.109 |
| asr_timings | 10.058 |
| ast_timings | 52.109 |
| describe_scenes | 19.573 |
| summarize_scenes | 9.108 |
| synthesize_synopsis | 8.946 |
| make_embedding | 6.541 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 75.942 |
| branch_yolo_total | 15.634 |
| branch_audio_total | 77.284 |
