# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 18:49:07 UTC | rXZfBpQq3I8_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 200.109 | 0.774 | 69.679 | 24.552 | 11.671 | 13.455 | 5.469 |

## 2026-06-26 18:49:07 UTC | rXZfBpQq3I8_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/rXZfBpQq3I8_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `200.109` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.774 |
| save_clips | - |
| sample_frames | 1.394 |
| caption_frames | 57.588 |
| sample_fps | 2.446 |
| detect_object_yolo | 11.624 |
| audio_scan | 16.150 |
| asr_timings | 11.380 |
| ast_timings | 42.140 |
| describe_scenes | 24.552 |
| summarize_scenes | 11.671 |
| synthesize_synopsis | 13.455 |
| make_embedding | 5.469 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 58.987 |
| branch_yolo_total | 14.075 |
| branch_audio_total | 69.679 |
