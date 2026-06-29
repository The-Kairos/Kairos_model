# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 00:16:49 UTC | u-UA8t2EVpA_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 183.576 | 0.676 | 64.732 | 14.819 | 9.368 | 12.466 | 5.359 |

## 2026-06-27 00:16:49 UTC | u-UA8t2EVpA_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/u-UA8t2EVpA_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `183.576` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.676 |
| save_clips | - |
| sample_frames | 1.461 |
| caption_frames | 59.493 |
| sample_fps | 2.322 |
| detect_object_yolo | 11.427 |
| audio_scan | 8.593 |
| asr_timings | 11.805 |
| ast_timings | 44.325 |
| describe_scenes | 14.819 |
| summarize_scenes | 9.368 |
| synthesize_synopsis | 12.466 |
| make_embedding | 5.359 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 60.960 |
| branch_yolo_total | 13.755 |
| branch_audio_total | 64.732 |
