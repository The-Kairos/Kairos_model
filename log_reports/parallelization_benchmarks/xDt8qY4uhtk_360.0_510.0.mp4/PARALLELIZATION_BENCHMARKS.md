# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 03:39:29 UTC | xDt8qY4uhtk_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 95.667 | 0.809 | 40.159 | 8.192 | 4.688 | 6.840 | 2.272 |

## 2026-06-27 03:39:29 UTC | xDt8qY4uhtk_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/xDt8qY4uhtk_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `95.667` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.809 |
| save_clips | - |
| sample_frames | 0.435 |
| caption_frames | 22.339 |
| sample_fps | 1.965 |
| detect_object_yolo | 6.544 |
| audio_scan | 15.083 |
| asr_timings | 9.049 |
| ast_timings | 16.018 |
| describe_scenes | 8.192 |
| summarize_scenes | 4.688 |
| synthesize_synopsis | 6.840 |
| make_embedding | 2.272 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 22.780 |
| branch_yolo_total | 8.515 |
| branch_audio_total | 40.159 |
