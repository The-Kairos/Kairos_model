# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-23 13:13:09 UTC | 1FQbzjvqr1w_660.0_810.0.mp4 | sequential | gemini | gemini-embedding-001 | 209.858 | 0.814 | 75.649 | 40.126 | 16.565 | 22.345 | 3.377 |
| 2026-06-27 14:51:59 UTC | 1FQbzjvqr1w_660.0_810.0.mp4 | sequential | gemini | gemini-embedding-001 | 133.854 | 0.819 | 49.453 | 9.877 | 10.395 | 8.703 | 3.408 |

## 2026-06-23 13:13:09 UTC | 1FQbzjvqr1w_660.0_810.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/1FQbzjvqr1w_660.0_810.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `209.858` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.814 |
| save_clips | - |
| sample_frames | 1.227 |
| caption_frames | 36.980 |
| sample_fps | 2.252 |
| detect_object_yolo | 9.085 |
| audio_scan | 7.550 |
| asr_timings | 41.203 |
| ast_timings | 26.887 |
| describe_scenes | 40.126 |
| summarize_scenes | 16.565 |
| synthesize_synopsis | 22.345 |
| make_embedding | 3.377 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 38.213 |
| branch_yolo_total | 11.343 |
| branch_audio_total | 75.649 |

## 2026-06-27 14:51:59 UTC | 1FQbzjvqr1w_660.0_810.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/1FQbzjvqr1w_660.0_810.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `133.854` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.819 |
| save_clips | - |
| sample_frames | 1.212 |
| caption_frames | 37.406 |
| sample_fps | 2.255 |
| detect_object_yolo | 8.908 |
| audio_scan | 7.519 |
| asr_timings | 14.681 |
| ast_timings | 27.245 |
| describe_scenes | 9.877 |
| summarize_scenes | 10.395 |
| synthesize_synopsis | 8.703 |
| make_embedding | 3.408 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 38.624 |
| branch_yolo_total | 11.169 |
| branch_audio_total | 49.453 |
