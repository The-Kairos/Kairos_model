# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-19 22:27:23 UTC | -_s0sXOfS3w_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 154.714 | 1.926 | 78.625 | 6.602 | 5.555 | 12.633 | 1.142 |
| 2026-06-21 09:03:48 UTC | -_s0sXOfS3w_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 0.059 | - | - | - | - | - | - |
| 2026-06-21 20:53:29 UTC | -_s0sXOfS3w_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 0.060 | - | - | - | - | - | - |
| 2026-06-22 11:39:26 UTC | -_s0sXOfS3w_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 179.582 | 1.958 | 85.177 | 16.535 | 8.524 | 14.381 | 3.035 |

## 2026-06-19 22:27:23 UTC | -_s0sXOfS3w_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/-_s0sXOfS3w_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `154.714` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.926 |
| save_clips | - |
| sample_frames | 3.218 |
| caption_frames | 29.194 |
| sample_fps | 6.238 |
| detect_object_yolo | 8.247 |
| audio_scan | 16.712 |
| asr_timings | 38.879 |
| ast_timings | 23.025 |
| describe_scenes | 6.602 |
| summarize_scenes | 5.555 |
| synthesize_synopsis | 12.633 |
| make_embedding | 1.142 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 32.418 |
| branch_yolo_total | 14.491 |
| branch_audio_total | 78.625 |

## 2026-06-21 09:03:48 UTC | -_s0sXOfS3w_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/-_s0sXOfS3w_510.0_660.0.mp4`
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

## 2026-06-21 20:53:29 UTC | -_s0sXOfS3w_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/-_s0sXOfS3w_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `0.060` sec

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

## 2026-06-22 11:39:26 UTC | -_s0sXOfS3w_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/-_s0sXOfS3w_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `179.582` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.958 |
| save_clips | - |
| sample_frames | 3.250 |
| caption_frames | 30.746 |
| sample_fps | 6.348 |
| detect_object_yolo | 8.245 |
| audio_scan | 16.715 |
| asr_timings | 45.126 |
| ast_timings | 23.326 |
| describe_scenes | 16.535 |
| summarize_scenes | 8.524 |
| synthesize_synopsis | 14.381 |
| make_embedding | 3.035 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 34.001 |
| branch_yolo_total | 14.599 |
| branch_audio_total | 85.177 |
