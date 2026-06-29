# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 19:15:04 UTC | UYp5bX4rgOQ_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 215.603 | 0.845 | 70.687 | 29.014 | 15.696 | 16.990 | 5.443 |

## 2026-06-25 19:15:04 UTC | UYp5bX4rgOQ_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/UYp5bX4rgOQ_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `215.603` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.845 |
| save_clips | - |
| sample_frames | 1.603 |
| caption_frames | 59.717 |
| sample_fps | 2.587 |
| detect_object_yolo | 11.617 |
| audio_scan | 16.043 |
| asr_timings | 10.148 |
| ast_timings | 44.487 |
| describe_scenes | 29.014 |
| summarize_scenes | 15.696 |
| synthesize_synopsis | 16.990 |
| make_embedding | 5.443 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 61.326 |
| branch_yolo_total | 14.210 |
| branch_audio_total | 70.687 |
