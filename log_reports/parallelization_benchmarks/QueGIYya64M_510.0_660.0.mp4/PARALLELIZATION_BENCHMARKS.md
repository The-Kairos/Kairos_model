# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 15:53:46 UTC | QueGIYya64M_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 225.444 | 0.772 | 91.116 | 23.432 | 16.565 | 19.528 | 5.105 |

## 2026-06-25 15:53:46 UTC | QueGIYya64M_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/QueGIYya64M_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `225.444` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.772 |
| save_clips | - |
| sample_frames | 1.528 |
| caption_frames | 52.730 |
| sample_fps | 2.471 |
| detect_object_yolo | 10.734 |
| audio_scan | 12.113 |
| asr_timings | 39.228 |
| ast_timings | 39.767 |
| describe_scenes | 23.432 |
| summarize_scenes | 16.565 |
| synthesize_synopsis | 19.528 |
| make_embedding | 5.105 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 54.263 |
| branch_yolo_total | 13.211 |
| branch_audio_total | 91.116 |
