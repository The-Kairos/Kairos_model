# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-21 21:10:21 UTC | 2I9-kvemtSU_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 123.993 | 0.762 | 40.843 | 7.112 | 6.291 | 24.740 | 2.793 |

## 2026-06-21 21:10:21 UTC | 2I9-kvemtSU_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/2I9-kvemtSU_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `123.993` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.762 |
| save_clips | - |
| sample_frames | 0.776 |
| caption_frames | 29.287 |
| sample_fps | 2.075 |
| detect_object_yolo | 7.913 |
| audio_scan | 7.512 |
| asr_timings | 12.007 |
| ast_timings | 21.316 |
| describe_scenes | 7.112 |
| summarize_scenes | 6.291 |
| synthesize_synopsis | 24.740 |
| make_embedding | 2.793 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 30.069 |
| branch_yolo_total | 9.995 |
| branch_audio_total | 40.843 |
