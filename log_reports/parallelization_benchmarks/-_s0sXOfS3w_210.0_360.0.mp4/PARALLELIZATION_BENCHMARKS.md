# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-19 21:21:53 UTC | -_s0sXOfS3w_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 0.773 | - | - | - | - | - | 0.705 |
| 2026-06-19 22:23:11 UTC | -_s0sXOfS3w_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 0.059 | - | - | - | - | - | 0.705 |
| 2026-06-21 09:03:45 UTC | -_s0sXOfS3w_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 0.058 | - | - | - | - | - | 0.705 |
| 2026-06-21 20:53:27 UTC | -_s0sXOfS3w_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 0.060 | - | - | - | - | - | 0.705 |
| 2026-06-22 11:34:19 UTC | -_s0sXOfS3w_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 139.346 | 1.885 | 65.285 | 10.233 | 11.032 | 21.943 | 1.814 |

## 2026-06-19 21:21:53 UTC | -_s0sXOfS3w_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/-_s0sXOfS3w_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `0.773` sec

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
| make_embedding | 0.705 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | - |
| branch_yolo_total | - |
| branch_audio_total | - |

## 2026-06-19 22:23:11 UTC | -_s0sXOfS3w_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/-_s0sXOfS3w_210.0_360.0.mp4`
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
| make_embedding | 0.705 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | - |
| branch_yolo_total | - |
| branch_audio_total | - |

## 2026-06-21 09:03:45 UTC | -_s0sXOfS3w_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/-_s0sXOfS3w_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `0.058` sec

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
| make_embedding | 0.705 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | - |
| branch_yolo_total | - |
| branch_audio_total | - |

## 2026-06-21 20:53:27 UTC | -_s0sXOfS3w_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/-_s0sXOfS3w_210.0_360.0.mp4`
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
| make_embedding | 0.705 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | - |
| branch_yolo_total | - |
| branch_audio_total | - |

## 2026-06-22 11:34:19 UTC | -_s0sXOfS3w_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/-_s0sXOfS3w_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `139.346` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.885 |
| save_clips | - |
| sample_frames | 0.970 |
| caption_frames | 13.075 |
| sample_fps | 5.377 |
| detect_object_yolo | 6.327 |
| audio_scan | 15.705 |
| asr_timings | 39.787 |
| ast_timings | 9.785 |
| describe_scenes | 10.233 |
| summarize_scenes | 11.032 |
| synthesize_synopsis | 21.943 |
| make_embedding | 1.814 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 14.051 |
| branch_yolo_total | 11.710 |
| branch_audio_total | 65.285 |
