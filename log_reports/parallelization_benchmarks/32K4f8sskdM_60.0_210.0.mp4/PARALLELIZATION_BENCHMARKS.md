# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-23 15:37:20 UTC | 32K4f8sskdM_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 237.085 | 0.644 | 78.184 | 33.752 | 39.814 | 31.489 | 3.297 |
| 2026-06-24 09:34:13 UTC | 32K4f8sskdM_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 205.641 | 0.660 | 93.532 | 20.171 | 16.649 | 19.899 | 3.444 |

## 2026-06-23 15:37:20 UTC | 32K4f8sskdM_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/32K4f8sskdM_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `237.085` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.644 |
| save_clips | - |
| sample_frames | 1.215 |
| caption_frames | 36.314 |
| sample_fps | 2.116 |
| detect_object_yolo | 8.875 |
| audio_scan | 15.895 |
| asr_timings | 35.455 |
| ast_timings | 26.825 |
| describe_scenes | 33.752 |
| summarize_scenes | 39.814 |
| synthesize_synopsis | 31.489 |
| make_embedding | 3.297 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 37.535 |
| branch_yolo_total | 10.996 |
| branch_audio_total | 78.184 |

## 2026-06-24 09:34:13 UTC | 32K4f8sskdM_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/32K4f8sskdM_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `205.641` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.660 |
| save_clips | - |
| sample_frames | 1.239 |
| caption_frames | 37.253 |
| sample_fps | 2.162 |
| detect_object_yolo | 9.167 |
| audio_scan | 16.165 |
| asr_timings | 50.488 |
| ast_timings | 26.870 |
| describe_scenes | 20.171 |
| summarize_scenes | 16.649 |
| synthesize_synopsis | 19.899 |
| make_embedding | 3.444 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 38.498 |
| branch_yolo_total | 11.334 |
| branch_audio_total | 93.532 |
