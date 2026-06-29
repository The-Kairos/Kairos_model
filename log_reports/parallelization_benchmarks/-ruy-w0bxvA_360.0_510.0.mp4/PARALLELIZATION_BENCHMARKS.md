# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 13:09:14 UTC | -ruy-w0bxvA_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 126.708 | 0.766 | 49.978 | 7.948 | 6.449 | 9.528 | 3.336 |

## 2026-06-27 13:09:14 UTC | -ruy-w0bxvA_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/-ruy-w0bxvA_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `126.708` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.766 |
| save_clips | - |
| sample_frames | 0.895 |
| caption_frames | 36.000 |
| sample_fps | 2.177 |
| detect_object_yolo | 8.230 |
| audio_scan | 12.814 |
| asr_timings | 10.489 |
| ast_timings | 26.667 |
| describe_scenes | 7.948 |
| summarize_scenes | 6.449 |
| synthesize_synopsis | 9.528 |
| make_embedding | 3.336 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 36.901 |
| branch_yolo_total | 10.413 |
| branch_audio_total | 49.978 |
