# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 01:32:21 UTC | GErRlbPkMmQ_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 156.528 | 0.781 | 58.025 | 13.917 | 11.206 | 12.854 | 3.821 |

## 2026-06-25 01:32:21 UTC | GErRlbPkMmQ_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/GErRlbPkMmQ_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `156.528` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.781 |
| save_clips | - |
| sample_frames | 1.377 |
| caption_frames | 41.592 |
| sample_fps | 2.410 |
| detect_object_yolo | 9.136 |
| audio_scan | 15.828 |
| asr_timings | 8.824 |
| ast_timings | 33.365 |
| describe_scenes | 13.917 |
| summarize_scenes | 11.206 |
| synthesize_synopsis | 12.854 |
| make_embedding | 3.821 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 42.975 |
| branch_yolo_total | 11.551 |
| branch_audio_total | 58.025 |
