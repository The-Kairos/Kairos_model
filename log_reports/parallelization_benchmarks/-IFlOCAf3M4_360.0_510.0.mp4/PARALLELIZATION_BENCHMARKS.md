# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 07:27:21 UTC | -IFlOCAf3M4_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 1703.233 | 0.781 | 1613.692 | 12.070 | 10.594 | 18.997 | 3.455 |

## 2026-06-24 07:27:21 UTC | -IFlOCAf3M4_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/-IFlOCAf3M4_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `1703.233` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.781 |
| save_clips | - |
| sample_frames | 1.274 |
| caption_frames | 30.458 |
| sample_fps | 2.227 |
| detect_object_yolo | 8.365 |
| audio_scan | 14.740 |
| asr_timings | 1572.925 |
| ast_timings | 26.018 |
| describe_scenes | 12.070 |
| summarize_scenes | 10.594 |
| synthesize_synopsis | 18.997 |
| make_embedding | 3.455 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 31.737 |
| branch_yolo_total | 10.597 |
| branch_audio_total | 1613.692 |
