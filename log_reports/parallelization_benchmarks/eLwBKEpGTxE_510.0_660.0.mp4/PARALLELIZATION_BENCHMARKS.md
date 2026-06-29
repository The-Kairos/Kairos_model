# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 03:45:17 UTC | eLwBKEpGTxE_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 163.899 | 0.777 | 64.267 | 16.732 | 6.372 | 11.200 | 4.136 |

## 2026-06-26 03:45:17 UTC | eLwBKEpGTxE_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/eLwBKEpGTxE_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `163.899` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.777 |
| save_clips | - |
| sample_frames | 1.340 |
| caption_frames | 45.192 |
| sample_fps | 2.407 |
| detect_object_yolo | 9.994 |
| audio_scan | 15.402 |
| asr_timings | 12.696 |
| ast_timings | 36.160 |
| describe_scenes | 16.732 |
| summarize_scenes | 6.372 |
| synthesize_synopsis | 11.200 |
| make_embedding | 4.136 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 46.538 |
| branch_yolo_total | 12.407 |
| branch_audio_total | 64.267 |
