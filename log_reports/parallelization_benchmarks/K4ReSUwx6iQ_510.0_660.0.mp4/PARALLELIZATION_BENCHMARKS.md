# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 05:52:26 UTC | K4ReSUwx6iQ_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 204.320 | 0.809 | 108.446 | 12.811 | 9.355 | 15.212 | 3.615 |

## 2026-06-25 05:52:26 UTC | K4ReSUwx6iQ_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/K4ReSUwx6iQ_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `204.320` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.809 |
| save_clips | - |
| sample_frames | 1.091 |
| caption_frames | 40.363 |
| sample_fps | 2.234 |
| detect_object_yolo | 8.996 |
| audio_scan | 12.802 |
| asr_timings | 65.358 |
| ast_timings | 30.278 |
| describe_scenes | 12.811 |
| summarize_scenes | 9.355 |
| synthesize_synopsis | 15.212 |
| make_embedding | 3.615 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 41.460 |
| branch_yolo_total | 11.236 |
| branch_audio_total | 108.446 |
