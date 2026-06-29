# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 01:41:02 UTC | boJssjt0HGk_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 193.312 | 0.689 | 71.585 | 14.581 | 12.521 | 9.349 | 5.812 |

## 2026-06-26 01:41:02 UTC | boJssjt0HGk_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/boJssjt0HGk_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `193.312` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.689 |
| save_clips | - |
| sample_frames | 1.515 |
| caption_frames | 61.619 |
| sample_fps | 2.413 |
| detect_object_yolo | 11.819 |
| audio_scan | 16.148 |
| asr_timings | 7.105 |
| ast_timings | 48.324 |
| describe_scenes | 14.581 |
| summarize_scenes | 12.521 |
| synthesize_synopsis | 9.349 |
| make_embedding | 5.812 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 63.139 |
| branch_yolo_total | 14.237 |
| branch_audio_total | 71.585 |
