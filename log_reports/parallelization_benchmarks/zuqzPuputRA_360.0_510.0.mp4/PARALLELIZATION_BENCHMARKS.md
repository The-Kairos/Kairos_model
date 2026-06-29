# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 06:21:19 UTC | zuqzPuputRA_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 94.400 | 0.776 | 41.098 | 5.883 | 5.367 | 7.731 | 2.253 |

## 2026-06-27 06:21:19 UTC | zuqzPuputRA_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/zuqzPuputRA_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `94.400` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.776 |
| save_clips | - |
| sample_frames | 0.477 |
| caption_frames | 20.438 |
| sample_fps | 1.962 |
| detect_object_yolo | 7.025 |
| audio_scan | 16.016 |
| asr_timings | 9.299 |
| ast_timings | 15.775 |
| describe_scenes | 5.883 |
| summarize_scenes | 5.367 |
| synthesize_synopsis | 7.731 |
| make_embedding | 2.253 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 20.921 |
| branch_yolo_total | 8.993 |
| branch_audio_total | 41.098 |
