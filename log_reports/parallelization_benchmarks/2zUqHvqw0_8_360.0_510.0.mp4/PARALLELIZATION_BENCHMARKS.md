# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-23 15:10:36 UTC | 2zUqHvqw0_8_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 177.591 | 0.637 | 53.888 | 14.651 | 19.484 | 36.978 | 3.421 |
| 2026-06-24 09:10:15 UTC | 2zUqHvqw0_8_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 163.559 | 0.645 | 55.692 | 16.464 | 18.495 | 20.011 | 3.347 |

## 2026-06-23 15:10:36 UTC | 2zUqHvqw0_8_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/2zUqHvqw0_8_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `177.591` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.637 |
| save_clips | - |
| sample_frames | 1.009 |
| caption_frames | 35.843 |
| sample_fps | 2.057 |
| detect_object_yolo | 8.252 |
| audio_scan | 14.778 |
| asr_timings | 12.335 |
| ast_timings | 26.768 |
| describe_scenes | 14.651 |
| summarize_scenes | 19.484 |
| synthesize_synopsis | 36.978 |
| make_embedding | 3.421 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 36.858 |
| branch_yolo_total | 10.314 |
| branch_audio_total | 53.888 |

## 2026-06-24 09:10:15 UTC | 2zUqHvqw0_8_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/2zUqHvqw0_8_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `163.559` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.645 |
| save_clips | - |
| sample_frames | 1.007 |
| caption_frames | 36.083 |
| sample_fps | 2.078 |
| detect_object_yolo | 8.360 |
| audio_scan | 14.904 |
| asr_timings | 13.824 |
| ast_timings | 26.956 |
| describe_scenes | 16.464 |
| summarize_scenes | 18.495 |
| synthesize_synopsis | 20.011 |
| make_embedding | 3.347 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 37.096 |
| branch_yolo_total | 10.444 |
| branch_audio_total | 55.692 |
