# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-19 22:49:11 UTC | 0U3-7Ey3siA_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 152.150 | 3.023 | 54.503 | 9.246 | 7.575 | 11.762 | 1.585 |
| 2026-06-21 09:03:57 UTC | 0U3-7Ey3siA_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 0.060 | - | - | - | - | - | - |
| 2026-06-21 20:53:44 UTC | 0U3-7Ey3siA_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 0.060 | - | - | - | - | - | - |
| 2026-06-22 12:18:39 UTC | 0U3-7Ey3siA_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 190.687 | 3.040 | 55.605 | 21.560 | 25.635 | 15.611 | 3.869 |

## 2026-06-19 22:49:11 UTC | 0U3-7Ey3siA_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/0U3-7Ey3siA_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `152.150` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 3.023 |
| save_clips | - |
| sample_frames | 4.539 |
| caption_frames | 40.088 |
| sample_fps | 9.290 |
| detect_object_yolo | 9.203 |
| audio_scan | 15.405 |
| asr_timings | 7.492 |
| ast_timings | 31.598 |
| describe_scenes | 9.246 |
| summarize_scenes | 7.575 |
| synthesize_synopsis | 11.762 |
| make_embedding | 1.585 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 44.632 |
| branch_yolo_total | 18.499 |
| branch_audio_total | 54.503 |

## 2026-06-21 09:03:57 UTC | 0U3-7Ey3siA_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/0U3-7Ey3siA_210.0_360.0.mp4`
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

## 2026-06-21 20:53:44 UTC | 0U3-7Ey3siA_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/0U3-7Ey3siA_210.0_360.0.mp4`
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

## 2026-06-22 12:18:39 UTC | 0U3-7Ey3siA_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/0U3-7Ey3siA_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `190.687` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 3.040 |
| save_clips | - |
| sample_frames | 4.614 |
| caption_frames | 40.757 |
| sample_fps | 9.219 |
| detect_object_yolo | 9.370 |
| audio_scan | 15.397 |
| asr_timings | 8.360 |
| ast_timings | 31.839 |
| describe_scenes | 21.560 |
| summarize_scenes | 25.635 |
| synthesize_synopsis | 15.611 |
| make_embedding | 3.869 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 45.377 |
| branch_yolo_total | 18.595 |
| branch_audio_total | 55.605 |
