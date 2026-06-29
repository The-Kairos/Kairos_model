# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-23 16:21:15 UTC | 3M7s6SupWyU_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 165.401 | 0.651 | 44.289 | 24.395 | 30.645 | 23.011 | 2.799 |
| 2026-06-24 10:16:51 UTC | 3M7s6SupWyU_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 143.175 | 0.664 | 44.656 | 14.254 | 12.451 | 28.392 | 2.746 |

## 2026-06-23 16:21:15 UTC | 3M7s6SupWyU_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/3M7s6SupWyU_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `165.401` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.651 |
| save_clips | - |
| sample_frames | 0.645 |
| caption_frames | 28.339 |
| sample_fps | 1.934 |
| detect_object_yolo | 7.328 |
| audio_scan | 12.680 |
| asr_timings | 10.746 |
| ast_timings | 20.854 |
| describe_scenes | 24.395 |
| summarize_scenes | 30.645 |
| synthesize_synopsis | 23.011 |
| make_embedding | 2.799 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 28.989 |
| branch_yolo_total | 9.267 |
| branch_audio_total | 44.289 |

## 2026-06-24 10:16:51 UTC | 3M7s6SupWyU_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/3M7s6SupWyU_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `143.175` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.664 |
| save_clips | - |
| sample_frames | 0.649 |
| caption_frames | 28.577 |
| sample_fps | 1.943 |
| detect_object_yolo | 7.455 |
| audio_scan | 12.793 |
| asr_timings | 11.120 |
| ast_timings | 20.733 |
| describe_scenes | 14.254 |
| summarize_scenes | 12.451 |
| synthesize_synopsis | 28.392 |
| make_embedding | 2.746 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 29.232 |
| branch_yolo_total | 9.404 |
| branch_audio_total | 44.656 |
