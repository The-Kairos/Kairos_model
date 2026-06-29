# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-23 15:13:58 UTC | 2zUqHvqw0_8_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 201.184 | 0.636 | 61.428 | 33.494 | 18.475 | 19.016 | 4.635 |
| 2026-06-24 09:13:36 UTC | 2zUqHvqw0_8_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 200.544 | 0.641 | 61.251 | 21.128 | 25.669 | 24.558 | 4.465 |

## 2026-06-23 15:13:58 UTC | 2zUqHvqw0_8_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/2zUqHvqw0_8_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `201.184` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.636 |
| save_clips | - |
| sample_frames | 1.614 |
| caption_frames | 48.672 |
| sample_fps | 2.272 |
| detect_object_yolo | 9.582 |
| audio_scan | 14.790 |
| asr_timings | 10.020 |
| ast_timings | 36.610 |
| describe_scenes | 33.494 |
| summarize_scenes | 18.475 |
| synthesize_synopsis | 19.016 |
| make_embedding | 4.635 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 50.292 |
| branch_yolo_total | 11.860 |
| branch_audio_total | 61.428 |

## 2026-06-24 09:13:36 UTC | 2zUqHvqw0_8_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/2zUqHvqw0_8_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `200.544` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.641 |
| save_clips | - |
| sample_frames | 1.617 |
| caption_frames | 47.861 |
| sample_fps | 2.289 |
| detect_object_yolo | 9.683 |
| audio_scan | 14.921 |
| asr_timings | 9.630 |
| ast_timings | 36.691 |
| describe_scenes | 21.128 |
| summarize_scenes | 25.669 |
| synthesize_synopsis | 24.558 |
| make_embedding | 4.465 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 49.483 |
| branch_yolo_total | 11.978 |
| branch_audio_total | 61.251 |
