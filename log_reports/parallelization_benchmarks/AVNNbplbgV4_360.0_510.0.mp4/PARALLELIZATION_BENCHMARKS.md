# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 19:19:01 UTC | AVNNbplbgV4_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 134.609 | 0.787 | 54.357 | 20.730 | 8.647 | 12.828 | 2.477 |

## 2026-06-24 19:19:01 UTC | AVNNbplbgV4_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/AVNNbplbgV4_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `134.609` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.787 |
| save_clips | - |
| sample_frames | 0.593 |
| caption_frames | 23.034 |
| sample_fps | 2.008 |
| detect_object_yolo | 7.675 |
| audio_scan | 12.989 |
| asr_timings | 25.407 |
| ast_timings | 15.952 |
| describe_scenes | 20.730 |
| summarize_scenes | 8.647 |
| synthesize_synopsis | 12.828 |
| make_embedding | 2.477 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 23.633 |
| branch_yolo_total | 9.689 |
| branch_audio_total | 54.357 |
