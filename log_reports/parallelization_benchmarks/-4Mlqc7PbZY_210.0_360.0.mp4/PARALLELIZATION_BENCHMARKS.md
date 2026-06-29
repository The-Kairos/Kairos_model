# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-19 21:21:45 UTC | -4Mlqc7PbZY_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 3.005 | - | - | - | - | - | 1.191 |
| 2026-06-19 22:23:06 UTC | -4Mlqc7PbZY_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 1.706 | - | - | - | - | - | 1.191 |
| 2026-06-21 09:03:41 UTC | -4Mlqc7PbZY_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 1.649 | - | - | - | - | - | 1.191 |
| 2026-06-21 20:53:23 UTC | -4Mlqc7PbZY_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 1.701 | - | - | - | - | - | 1.191 |
| 2026-06-22 11:23:56 UTC | -4Mlqc7PbZY_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 139.054 | 3.405 | 47.337 | 12.749 | 9.098 | 19.539 | 2.695 |

## 2026-06-19 21:21:45 UTC | -4Mlqc7PbZY_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/-4Mlqc7PbZY_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `3.005` sec

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
| make_embedding | 1.191 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | - |
| branch_yolo_total | - |
| branch_audio_total | - |

## 2026-06-19 22:23:06 UTC | -4Mlqc7PbZY_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/-4Mlqc7PbZY_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `1.706` sec

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
| make_embedding | 1.191 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | - |
| branch_yolo_total | - |
| branch_audio_total | - |

## 2026-06-21 09:03:41 UTC | -4Mlqc7PbZY_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/-4Mlqc7PbZY_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `1.649` sec

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
| make_embedding | 1.191 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | - |
| branch_yolo_total | - |
| branch_audio_total | - |

## 2026-06-21 20:53:23 UTC | -4Mlqc7PbZY_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/-4Mlqc7PbZY_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `1.701` sec

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
| make_embedding | 1.191 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | - |
| branch_yolo_total | - |
| branch_audio_total | - |

## 2026-06-22 11:23:56 UTC | -4Mlqc7PbZY_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/-4Mlqc7PbZY_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `139.054` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 3.405 |
| save_clips | - |
| sample_frames | 3.387 |
| caption_frames | 17.375 |
| sample_fps | 10.518 |
| detect_object_yolo | 6.792 |
| audio_scan | 18.337 |
| asr_timings | 11.518 |
| ast_timings | 17.475 |
| describe_scenes | 12.749 |
| summarize_scenes | 9.098 |
| synthesize_synopsis | 19.539 |
| make_embedding | 2.695 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 20.767 |
| branch_yolo_total | 17.315 |
| branch_audio_total | 47.337 |
