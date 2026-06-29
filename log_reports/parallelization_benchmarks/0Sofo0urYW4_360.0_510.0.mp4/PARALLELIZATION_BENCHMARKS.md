# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 13:41:24 UTC | 0Sofo0urYW4_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 177.020 | 0.811 | 66.523 | 12.645 | 10.172 | 7.661 | 5.397 |

## 2026-06-27 13:41:24 UTC | 0Sofo0urYW4_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/0Sofo0urYW4_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `177.020` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.811 |
| save_clips | - |
| sample_frames | 1.425 |
| caption_frames | 57.326 |
| sample_fps | 2.536 |
| detect_object_yolo | 11.121 |
| audio_scan | 13.898 |
| asr_timings | 8.386 |
| ast_timings | 44.230 |
| describe_scenes | 12.645 |
| summarize_scenes | 10.172 |
| synthesize_synopsis | 7.661 |
| make_embedding | 5.397 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 58.757 |
| branch_yolo_total | 13.663 |
| branch_audio_total | 66.523 |
