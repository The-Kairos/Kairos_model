# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 12:11:17 UTC | jqt8j8h_U_8_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 77.489 | 0.733 | 21.645 | 7.222 | 8.420 | 16.073 | 1.544 |

## 2026-06-26 12:11:17 UTC | jqt8j8h_U_8_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/jqt8j8h_U_8_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `77.489` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.733 |
| save_clips | - |
| sample_frames | 0.299 |
| caption_frames | 12.122 |
| sample_fps | 1.708 |
| detect_object_yolo | 6.323 |
| audio_scan | 6.525 |
| asr_timings | 7.663 |
| ast_timings | 7.449 |
| describe_scenes | 7.222 |
| summarize_scenes | 8.420 |
| synthesize_synopsis | 16.073 |
| make_embedding | 1.544 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 12.427 |
| branch_yolo_total | 8.038 |
| branch_audio_total | 21.645 |
