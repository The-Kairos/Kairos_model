# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-21 10:02:55 UTC | 2vFzBa_WKNg_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 168.398 | 3.532 | 65.919 | 10.175 | 7.795 | 9.579 | 3.912 |
| 2026-06-21 21:42:20 UTC | 2vFzBa_WKNg_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 167.970 | 3.615 | 66.988 | 9.823 | 7.366 | 6.964 | 3.887 |

## 2026-06-21 10:02:55 UTC | 2vFzBa_WKNg_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/2vFzBa_WKNg_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `168.398` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 3.532 |
| save_clips | - |
| sample_frames | 4.732 |
| caption_frames | 41.855 |
| sample_fps | 10.971 |
| detect_object_yolo | 8.606 |
| audio_scan | 12.810 |
| asr_timings | 21.607 |
| ast_timings | 31.493 |
| describe_scenes | 10.175 |
| summarize_scenes | 7.795 |
| synthesize_synopsis | 9.579 |
| make_embedding | 3.912 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 46.593 |
| branch_yolo_total | 19.582 |
| branch_audio_total | 65.919 |

## 2026-06-21 21:42:20 UTC | 2vFzBa_WKNg_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/2vFzBa_WKNg_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `167.970` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 3.615 |
| save_clips | - |
| sample_frames | 4.895 |
| caption_frames | 43.048 |
| sample_fps | 10.996 |
| detect_object_yolo | 8.982 |
| audio_scan | 12.954 |
| asr_timings | 21.939 |
| ast_timings | 32.087 |
| describe_scenes | 9.823 |
| summarize_scenes | 7.366 |
| synthesize_synopsis | 6.964 |
| make_embedding | 3.887 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 47.949 |
| branch_yolo_total | 19.984 |
| branch_audio_total | 66.988 |
