# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-23 14:26:52 UTC | 2ALPz7TU1WY_660.0_810.0.mp4 | sequential | gemini | gemini-embedding-001 | 224.026 | 0.770 | 58.160 | 22.524 | 20.346 | 57.007 | 4.252 |
| 2026-06-27 15:43:40 UTC | 2ALPz7TU1WY_660.0_810.0.mp4 | sequential | gemini | gemini-embedding-001 | 174.551 | 0.774 | 59.407 | 9.737 | 14.872 | 7.543 | 20.477 |

## 2026-06-23 14:26:52 UTC | 2ALPz7TU1WY_660.0_810.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/2ALPz7TU1WY_660.0_810.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `224.026` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.770 |
| save_clips | - |
| sample_frames | 1.080 |
| caption_frames | 46.248 |
| sample_fps | 2.275 |
| detect_object_yolo | 9.919 |
| audio_scan | 11.734 |
| asr_timings | 10.714 |
| ast_timings | 35.705 |
| describe_scenes | 22.524 |
| summarize_scenes | 20.346 |
| synthesize_synopsis | 57.007 |
| make_embedding | 4.252 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 47.334 |
| branch_yolo_total | 12.199 |
| branch_audio_total | 58.160 |

## 2026-06-27 15:43:40 UTC | 2ALPz7TU1WY_660.0_810.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/2ALPz7TU1WY_660.0_810.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `174.551` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.774 |
| save_clips | - |
| sample_frames | 1.086 |
| caption_frames | 47.235 |
| sample_fps | 2.263 |
| detect_object_yolo | 9.747 |
| audio_scan | 11.831 |
| asr_timings | 11.778 |
| ast_timings | 35.789 |
| describe_scenes | 9.737 |
| summarize_scenes | 14.872 |
| synthesize_synopsis | 7.543 |
| make_embedding | 20.477 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 48.327 |
| branch_yolo_total | 12.016 |
| branch_audio_total | 59.407 |
