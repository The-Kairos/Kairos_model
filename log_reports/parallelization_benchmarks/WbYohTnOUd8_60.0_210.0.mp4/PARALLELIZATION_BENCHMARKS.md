# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 20:50:05 UTC | WbYohTnOUd8_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 197.730 | 0.670 | 81.312 | 17.322 | 8.356 | 24.670 | 4.215 |

## 2026-06-25 20:50:05 UTC | WbYohTnOUd8_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/WbYohTnOUd8_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `197.730` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.670 |
| save_clips | - |
| sample_frames | 1.305 |
| caption_frames | 46.258 |
| sample_fps | 2.200 |
| detect_object_yolo | 10.022 |
| audio_scan | 8.556 |
| asr_timings | 37.385 |
| ast_timings | 35.364 |
| describe_scenes | 17.322 |
| summarize_scenes | 8.356 |
| synthesize_synopsis | 24.670 |
| make_embedding | 4.215 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 47.569 |
| branch_yolo_total | 12.228 |
| branch_audio_total | 81.312 |
