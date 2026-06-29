# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 10:32:46 UTC | jF9fQEliENc_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 124.816 | 0.641 | 44.159 | 8.888 | 16.402 | 21.937 | 2.273 |

## 2026-06-26 10:32:46 UTC | jF9fQEliENc_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/jF9fQEliENc_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `124.816` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.641 |
| save_clips | - |
| sample_frames | 0.404 |
| caption_frames | 20.396 |
| sample_fps | 1.769 |
| detect_object_yolo | 6.516 |
| audio_scan | 16.154 |
| asr_timings | 12.380 |
| ast_timings | 15.617 |
| describe_scenes | 8.888 |
| summarize_scenes | 16.402 |
| synthesize_synopsis | 21.937 |
| make_embedding | 2.273 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 20.806 |
| branch_yolo_total | 8.290 |
| branch_audio_total | 44.159 |
