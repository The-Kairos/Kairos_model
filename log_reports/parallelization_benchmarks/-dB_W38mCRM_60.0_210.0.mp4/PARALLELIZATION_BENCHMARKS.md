# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-19 22:37:38 UTC | -dB_W38mCRM_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 179.822 | 1.437 | 66.862 | 10.624 | 12.263 | 14.668 | 2.085 |
| 2026-06-21 09:03:51 UTC | -dB_W38mCRM_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 0.059 | - | - | - | - | - | - |
| 2026-06-21 20:53:34 UTC | -dB_W38mCRM_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 0.060 | - | - | - | - | - | - |
| 2026-06-22 11:53:12 UTC | -dB_W38mCRM_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 202.151 | 1.469 | 68.550 | 18.290 | 18.757 | 15.141 | 5.628 |

## 2026-06-19 22:37:38 UTC | -dB_W38mCRM_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/-dB_W38mCRM_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `179.822` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.437 |
| save_clips | - |
| sample_frames | 3.291 |
| caption_frames | 50.855 |
| sample_fps | 5.866 |
| detect_object_yolo | 10.513 |
| audio_scan | 14.441 |
| asr_timings | 12.182 |
| ast_timings | 40.231 |
| describe_scenes | 10.624 |
| summarize_scenes | 12.263 |
| synthesize_synopsis | 14.668 |
| make_embedding | 2.085 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 54.152 |
| branch_yolo_total | 16.385 |
| branch_audio_total | 66.862 |

## 2026-06-21 09:03:51 UTC | -dB_W38mCRM_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/-dB_W38mCRM_60.0_210.0.mp4`
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

## 2026-06-21 20:53:34 UTC | -dB_W38mCRM_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/-dB_W38mCRM_60.0_210.0.mp4`
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

## 2026-06-22 11:53:12 UTC | -dB_W38mCRM_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/-dB_W38mCRM_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `202.151` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.469 |
| save_clips | - |
| sample_frames | 3.328 |
| caption_frames | 52.906 |
| sample_fps | 5.883 |
| detect_object_yolo | 10.801 |
| audio_scan | 14.618 |
| asr_timings | 13.048 |
| ast_timings | 40.876 |
| describe_scenes | 18.290 |
| summarize_scenes | 18.757 |
| synthesize_synopsis | 15.141 |
| make_embedding | 5.628 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 56.241 |
| branch_yolo_total | 16.691 |
| branch_audio_total | 68.550 |
