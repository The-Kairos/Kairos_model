# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-21 20:53:46 UTC | 0U3-7Ey3siA_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 0.060 | - | - | - | - | - | - |
| 2026-06-22 12:25:37 UTC | 0U3-7Ey3siA_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 196.188 | 0.649 | 57.454 | 20.672 | 19.331 | 36.805 | 3.884 |

## 2026-06-21 20:53:46 UTC | 0U3-7Ey3siA_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/0U3-7Ey3siA_60.0_210.0.mp4`
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

## 2026-06-22 12:25:37 UTC | 0U3-7Ey3siA_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/0U3-7Ey3siA_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `196.188` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.649 |
| save_clips | - |
| sample_frames | 1.037 |
| caption_frames | 43.352 |
| sample_fps | 2.108 |
| detect_object_yolo | 9.493 |
| audio_scan | 15.844 |
| asr_timings | 9.092 |
| ast_timings | 32.510 |
| describe_scenes | 20.672 |
| summarize_scenes | 19.331 |
| synthesize_synopsis | 36.805 |
| make_embedding | 3.884 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 44.395 |
| branch_yolo_total | 11.607 |
| branch_audio_total | 57.454 |
