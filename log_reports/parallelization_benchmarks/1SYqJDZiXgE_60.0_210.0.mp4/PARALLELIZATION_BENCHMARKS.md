# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-23 13:35:04 UTC | 1SYqJDZiXgE_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 202.239 | 0.656 | 58.820 | 30.420 | 16.914 | 31.365 | 4.200 |
| 2026-06-27 15:07:55 UTC | 1SYqJDZiXgE_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 161.345 | 0.639 | 62.050 | 17.267 | 7.454 | 8.496 | 4.220 |

## 2026-06-23 13:35:04 UTC | 1SYqJDZiXgE_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/1SYqJDZiXgE_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `202.239` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.656 |
| save_clips | - |
| sample_frames | 1.296 |
| caption_frames | 45.596 |
| sample_fps | 2.142 |
| detect_object_yolo | 9.447 |
| audio_scan | 8.480 |
| asr_timings | 14.965 |
| ast_timings | 35.368 |
| describe_scenes | 30.420 |
| summarize_scenes | 16.914 |
| synthesize_synopsis | 31.365 |
| make_embedding | 4.200 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 46.898 |
| branch_yolo_total | 11.595 |
| branch_audio_total | 58.820 |

## 2026-06-27 15:07:55 UTC | 1SYqJDZiXgE_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/1SYqJDZiXgE_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `161.345` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.639 |
| save_clips | - |
| sample_frames | 1.312 |
| caption_frames | 46.102 |
| sample_fps | 2.241 |
| detect_object_yolo | 10.077 |
| audio_scan | 8.704 |
| asr_timings | 17.445 |
| ast_timings | 35.893 |
| describe_scenes | 17.267 |
| summarize_scenes | 7.454 |
| synthesize_synopsis | 8.496 |
| make_embedding | 4.220 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 47.420 |
| branch_yolo_total | 12.324 |
| branch_audio_total | 62.050 |
