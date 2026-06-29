# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-23 16:09:57 UTC | 3LSElHQKRho_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 241.573 | 0.793 | 72.108 | 28.638 | 21.537 | 40.204 | 5.367 |
| 2026-06-24 10:06:29 UTC | 3LSElHQKRho_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 225.029 | 0.807 | 68.105 | 24.668 | 34.257 | 20.055 | 5.358 |

## 2026-06-23 16:09:57 UTC | 3LSElHQKRho_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/3LSElHQKRho_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `241.573` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.793 |
| save_clips | - |
| sample_frames | 1.547 |
| caption_frames | 56.317 |
| sample_fps | 2.604 |
| detect_object_yolo | 11.066 |
| audio_scan | 10.653 |
| asr_timings | 18.552 |
| ast_timings | 42.895 |
| describe_scenes | 28.638 |
| summarize_scenes | 21.537 |
| synthesize_synopsis | 40.204 |
| make_embedding | 5.367 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 57.869 |
| branch_yolo_total | 13.675 |
| branch_audio_total | 72.108 |

## 2026-06-24 10:06:29 UTC | 3LSElHQKRho_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/3LSElHQKRho_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `225.029` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.807 |
| save_clips | - |
| sample_frames | 1.539 |
| caption_frames | 55.198 |
| sample_fps | 2.619 |
| detect_object_yolo | 11.031 |
| audio_scan | 10.727 |
| asr_timings | 14.073 |
| ast_timings | 43.297 |
| describe_scenes | 24.668 |
| summarize_scenes | 34.257 |
| synthesize_synopsis | 20.055 |
| make_embedding | 5.358 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 56.743 |
| branch_yolo_total | 13.656 |
| branch_audio_total | 68.105 |
