# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 10:41:11 UTC | jF9fQEliENc_660.0_810.0.mp4 | sequential | gemini | gemini-embedding-001 | 143.562 | 0.661 | 41.951 | 16.691 | 12.016 | 24.553 | 3.030 |

## 2026-06-26 10:41:11 UTC | jF9fQEliENc_660.0_810.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/jF9fQEliENc_660.0_810.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `143.562` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.661 |
| save_clips | - |
| sample_frames | 0.896 |
| caption_frames | 32.460 |
| sample_fps | 2.041 |
| detect_object_yolo | 7.822 |
| audio_scan | 7.604 |
| asr_timings | 10.069 |
| ast_timings | 24.270 |
| describe_scenes | 16.691 |
| summarize_scenes | 12.016 |
| synthesize_synopsis | 24.553 |
| make_embedding | 3.030 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 33.362 |
| branch_yolo_total | 9.869 |
| branch_audio_total | 41.951 |
