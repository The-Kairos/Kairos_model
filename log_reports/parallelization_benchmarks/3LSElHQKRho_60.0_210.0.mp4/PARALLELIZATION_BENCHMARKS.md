# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-23 16:13:39 UTC | 3LSElHQKRho_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 220.822 | 0.809 | 65.178 | 30.394 | 17.019 | 29.202 | 5.367 |
| 2026-06-24 10:09:49 UTC | 3LSElHQKRho_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 199.267 | 0.825 | 65.398 | 19.924 | 14.491 | 19.206 | 5.473 |

## 2026-06-23 16:13:39 UTC | 3LSElHQKRho_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/3LSElHQKRho_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `220.822` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.809 |
| save_clips | - |
| sample_frames | 1.669 |
| caption_frames | 56.383 |
| sample_fps | 2.617 |
| detect_object_yolo | 10.795 |
| audio_scan | 9.540 |
| asr_timings | 12.044 |
| ast_timings | 43.586 |
| describe_scenes | 30.394 |
| summarize_scenes | 17.019 |
| synthesize_synopsis | 29.202 |
| make_embedding | 5.367 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 58.058 |
| branch_yolo_total | 13.418 |
| branch_audio_total | 65.178 |

## 2026-06-24 10:09:49 UTC | 3LSElHQKRho_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/3LSElHQKRho_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `199.267` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.825 |
| save_clips | - |
| sample_frames | 1.673 |
| caption_frames | 57.364 |
| sample_fps | 2.650 |
| detect_object_yolo | 10.878 |
| audio_scan | 9.620 |
| asr_timings | 11.989 |
| ast_timings | 43.781 |
| describe_scenes | 19.924 |
| summarize_scenes | 14.491 |
| synthesize_synopsis | 19.206 |
| make_embedding | 5.473 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 59.043 |
| branch_yolo_total | 13.533 |
| branch_audio_total | 65.398 |
