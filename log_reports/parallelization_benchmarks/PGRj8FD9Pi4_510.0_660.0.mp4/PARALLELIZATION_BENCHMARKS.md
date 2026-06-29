# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 13:05:11 UTC | PGRj8FD9Pi4_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 240.811 | 0.635 | 66.470 | 41.228 | 22.234 | 31.888 | 5.503 |

## 2026-06-25 13:05:11 UTC | PGRj8FD9Pi4_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/PGRj8FD9Pi4_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `240.811` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.635 |
| save_clips | - |
| sample_frames | 1.295 |
| caption_frames | 56.983 |
| sample_fps | 2.293 |
| detect_object_yolo | 10.886 |
| audio_scan | 8.780 |
| asr_timings | 13.238 |
| ast_timings | 44.444 |
| describe_scenes | 41.228 |
| summarize_scenes | 22.234 |
| synthesize_synopsis | 31.888 |
| make_embedding | 5.503 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 58.284 |
| branch_yolo_total | 13.185 |
| branch_audio_total | 66.470 |
