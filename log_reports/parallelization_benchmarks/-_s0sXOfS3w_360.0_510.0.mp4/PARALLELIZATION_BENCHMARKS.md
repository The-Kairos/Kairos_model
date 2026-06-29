# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-19 22:24:47 UTC | -_s0sXOfS3w_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 94.973 | 1.908 | 54.187 | 6.006 | 2.509 | 5.600 | 0.951 |
| 2026-06-21 09:03:47 UTC | -_s0sXOfS3w_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 0.059 | - | - | - | - | - | - |
| 2026-06-21 20:53:28 UTC | -_s0sXOfS3w_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 0.059 | - | - | - | - | - | - |
| 2026-06-22 11:36:25 UTC | -_s0sXOfS3w_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 124.806 | 1.884 | 63.505 | 7.012 | 4.711 | 18.777 | 1.834 |

## 2026-06-19 22:24:47 UTC | -_s0sXOfS3w_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/-_s0sXOfS3w_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `94.973` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.908 |
| save_clips | - |
| sample_frames | 0.885 |
| caption_frames | 10.355 |
| sample_fps | 5.212 |
| detect_object_yolo | 6.113 |
| audio_scan | 16.784 |
| asr_timings | 27.824 |
| ast_timings | 9.571 |
| describe_scenes | 6.006 |
| summarize_scenes | 2.509 |
| synthesize_synopsis | 5.600 |
| make_embedding | 0.951 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 11.245 |
| branch_yolo_total | 11.330 |
| branch_audio_total | 54.187 |

## 2026-06-21 09:03:47 UTC | -_s0sXOfS3w_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/-_s0sXOfS3w_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `0.059` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | - |
| save_clips | - |
| sample_frames | - |
| caption_frames | - |
| sample_fps | - |
| detect_object_yolo | - |
| audio_scan | - |
| asr_timings | - |
| ast_timings | - |
| describe_scenes | - |
| summarize_scenes | - |
| synthesize_synopsis | - |
| make_embedding | - |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | - |
| branch_yolo_total | - |
| branch_audio_total | - |

## 2026-06-21 20:53:28 UTC | -_s0sXOfS3w_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/-_s0sXOfS3w_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `0.059` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | - |
| save_clips | - |
| sample_frames | - |
| caption_frames | - |
| sample_fps | - |
| detect_object_yolo | - |
| audio_scan | - |
| asr_timings | - |
| ast_timings | - |
| describe_scenes | - |
| summarize_scenes | - |
| synthesize_synopsis | - |
| make_embedding | - |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | - |
| branch_yolo_total | - |
| branch_audio_total | - |

## 2026-06-22 11:36:25 UTC | -_s0sXOfS3w_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/-_s0sXOfS3w_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `124.806` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.884 |
| save_clips | - |
| sample_frames | 0.869 |
| caption_frames | 13.134 |
| sample_fps | 5.198 |
| detect_object_yolo | 6.498 |
| audio_scan | 15.649 |
| asr_timings | 37.962 |
| ast_timings | 9.884 |
| describe_scenes | 7.012 |
| summarize_scenes | 4.711 |
| synthesize_synopsis | 18.777 |
| make_embedding | 1.834 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 14.009 |
| branch_yolo_total | 11.701 |
| branch_audio_total | 63.505 |
