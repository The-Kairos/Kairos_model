# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-19 22:31:01 UTC | -dB_W38mCRM_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 217.263 | 1.374 | 85.233 | 15.679 | 10.402 | 7.899 | 2.787 |
| 2026-06-21 09:03:49 UTC | -dB_W38mCRM_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 0.059 | - | - | - | - | - | - |
| 2026-06-21 20:53:31 UTC | -dB_W38mCRM_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 0.061 | - | - | - | - | - | - |
| 2026-06-22 11:45:53 UTC | -dB_W38mCRM_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 244.767 | 1.349 | 86.434 | 23.448 | 12.421 | 18.037 | 6.550 |

## 2026-06-19 22:31:01 UTC | -dB_W38mCRM_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/-dB_W38mCRM_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `217.263` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.374 |
| save_clips | - |
| sample_frames | 3.555 |
| caption_frames | 70.000 |
| sample_fps | 5.868 |
| detect_object_yolo | 13.133 |
| audio_scan | 16.536 |
| asr_timings | 15.417 |
| ast_timings | 53.271 |
| describe_scenes | 15.679 |
| summarize_scenes | 10.402 |
| synthesize_synopsis | 7.899 |
| make_embedding | 2.787 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 73.561 |
| branch_yolo_total | 19.006 |
| branch_audio_total | 85.233 |

## 2026-06-21 09:03:49 UTC | -dB_W38mCRM_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/-dB_W38mCRM_210.0_360.0.mp4`
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

## 2026-06-21 20:53:31 UTC | -dB_W38mCRM_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/-dB_W38mCRM_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `0.061` sec

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

## 2026-06-22 11:45:53 UTC | -dB_W38mCRM_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/-dB_W38mCRM_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `244.767` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.349 |
| save_clips | - |
| sample_frames | 3.578 |
| caption_frames | 72.244 |
| sample_fps | 5.966 |
| detect_object_yolo | 13.363 |
| audio_scan | 16.685 |
| asr_timings | 15.975 |
| ast_timings | 53.765 |
| describe_scenes | 23.448 |
| summarize_scenes | 12.421 |
| synthesize_synopsis | 18.037 |
| make_embedding | 6.550 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 75.828 |
| branch_yolo_total | 19.335 |
| branch_audio_total | 86.434 |
