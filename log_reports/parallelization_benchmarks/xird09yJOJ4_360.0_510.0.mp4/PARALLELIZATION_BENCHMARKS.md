# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 03:57:52 UTC | xird09yJOJ4_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 78.231 | 0.774 | 35.297 | 4.491 | 4.434 | 7.467 | 1.779 |

## 2026-06-27 03:57:52 UTC | xird09yJOJ4_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/xird09yJOJ4_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `78.231` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.774 |
| save_clips | - |
| sample_frames | 0.267 |
| caption_frames | 13.926 |
| sample_fps | 1.873 |
| detect_object_yolo | 6.490 |
| audio_scan | 15.295 |
| asr_timings | 10.096 |
| ast_timings | 9.897 |
| describe_scenes | 4.491 |
| summarize_scenes | 4.434 |
| synthesize_synopsis | 7.467 |
| make_embedding | 1.779 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 14.199 |
| branch_yolo_total | 8.369 |
| branch_audio_total | 35.297 |
