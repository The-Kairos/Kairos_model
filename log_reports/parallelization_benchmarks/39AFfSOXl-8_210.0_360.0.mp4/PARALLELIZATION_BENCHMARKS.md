# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-21 10:08:09 UTC | 39AFfSOXl-8_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 134.180 | 1.136 | 49.297 | 7.208 | 12.970 | 7.707 | 3.373 |
| 2026-06-21 21:47:30 UTC | 39AFfSOXl-8_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 128.395 | 1.141 | 49.083 | 7.526 | 5.959 | 7.066 | 3.337 |

## 2026-06-21 10:08:09 UTC | 39AFfSOXl-8_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/39AFfSOXl-8_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `134.180` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.136 |
| save_clips | - |
| sample_frames | 1.609 |
| caption_frames | 36.736 |
| sample_fps | 3.748 |
| detect_object_yolo | 9.094 |
| audio_scan | 13.781 |
| asr_timings | 8.605 |
| ast_timings | 26.903 |
| describe_scenes | 7.208 |
| summarize_scenes | 12.970 |
| synthesize_synopsis | 7.707 |
| make_embedding | 3.373 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 38.350 |
| branch_yolo_total | 12.847 |
| branch_audio_total | 49.297 |

## 2026-06-21 21:47:30 UTC | 39AFfSOXl-8_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/39AFfSOXl-8_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `128.395` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.141 |
| save_clips | - |
| sample_frames | 1.629 |
| caption_frames | 38.071 |
| sample_fps | 3.753 |
| detect_object_yolo | 9.426 |
| audio_scan | 13.862 |
| asr_timings | 8.138 |
| ast_timings | 27.074 |
| describe_scenes | 7.526 |
| summarize_scenes | 5.959 |
| synthesize_synopsis | 7.066 |
| make_embedding | 3.337 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 39.706 |
| branch_yolo_total | 13.185 |
| branch_audio_total | 49.083 |
