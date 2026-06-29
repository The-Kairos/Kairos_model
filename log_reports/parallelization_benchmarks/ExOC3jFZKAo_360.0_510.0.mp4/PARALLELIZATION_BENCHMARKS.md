# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 00:10:04 UTC | ExOC3jFZKAo_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 208.939 | 0.858 | 68.137 | 17.100 | 23.154 | 14.395 | 6.428 |

## 2026-06-25 00:10:04 UTC | ExOC3jFZKAo_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/ExOC3jFZKAo_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `208.939` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.858 |
| save_clips | - |
| sample_frames | 1.687 |
| caption_frames | 61.311 |
| sample_fps | 2.649 |
| detect_object_yolo | 11.790 |
| audio_scan | 11.850 |
| asr_timings | 10.394 |
| ast_timings | 45.885 |
| describe_scenes | 17.100 |
| summarize_scenes | 23.154 |
| synthesize_synopsis | 14.395 |
| make_embedding | 6.428 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 63.004 |
| branch_yolo_total | 14.445 |
| branch_audio_total | 68.137 |
