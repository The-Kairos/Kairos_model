# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-19 21:21:49 UTC | -4Mlqc7PbZY_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 0.988 | - | - | - | - | - | 0.913 |
| 2026-06-19 22:23:09 UTC | -4Mlqc7PbZY_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 0.059 | - | - | - | - | - | 0.913 |
| 2026-06-21 09:03:43 UTC | -4Mlqc7PbZY_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 0.059 | - | - | - | - | - | 0.913 |
| 2026-06-21 20:53:25 UTC | -4Mlqc7PbZY_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 0.061 | - | - | - | - | - | 0.913 |
| 2026-06-22 11:29:11 UTC | -4Mlqc7PbZY_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 150.450 | 3.215 | 51.116 | 14.762 | 12.741 | 17.885 | 2.624 |

## 2026-06-19 21:21:49 UTC | -4Mlqc7PbZY_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/-4Mlqc7PbZY_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `0.988` sec

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
| make_embedding | 0.913 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | - |
| branch_yolo_total | - |
| branch_audio_total | - |

## 2026-06-19 22:23:09 UTC | -4Mlqc7PbZY_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/-4Mlqc7PbZY_510.0_660.0.mp4`
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
| make_embedding | 0.913 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | - |
| branch_yolo_total | - |
| branch_audio_total | - |

## 2026-06-21 09:03:43 UTC | -4Mlqc7PbZY_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/-4Mlqc7PbZY_510.0_660.0.mp4`
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
| make_embedding | 0.913 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | - |
| branch_yolo_total | - |
| branch_audio_total | - |

## 2026-06-21 20:53:25 UTC | -4Mlqc7PbZY_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/-4Mlqc7PbZY_510.0_660.0.mp4`
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
| make_embedding | 0.913 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | - |
| branch_yolo_total | - |
| branch_audio_total | - |

## 2026-06-22 11:29:11 UTC | -4Mlqc7PbZY_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/-4Mlqc7PbZY_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `150.450` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 3.215 |
| save_clips | - |
| sample_frames | 2.907 |
| caption_frames | 26.136 |
| sample_fps | 10.002 |
| detect_object_yolo | 7.690 |
| audio_scan | 14.914 |
| asr_timings | 17.858 |
| ast_timings | 18.335 |
| describe_scenes | 14.762 |
| summarize_scenes | 12.741 |
| synthesize_synopsis | 17.885 |
| make_embedding | 2.624 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 29.049 |
| branch_yolo_total | 17.698 |
| branch_audio_total | 51.116 |
