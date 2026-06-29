# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 12:54:02 UTC | -eQfg7HJuKc_660.0_810.0.mp4 | sequential | gemini | gemini-embedding-001 | 160.171 | 0.801 | 56.451 | 11.590 | 8.999 | 15.436 | 4.180 |

## 2026-06-27 12:54:02 UTC | -eQfg7HJuKc_660.0_810.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/-eQfg7HJuKc_660.0_810.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `160.171` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.801 |
| save_clips | - |
| sample_frames | 1.593 |
| caption_frames | 47.451 |
| sample_fps | 2.519 |
| detect_object_yolo | 9.748 |
| audio_scan | 12.873 |
| asr_timings | 8.979 |
| ast_timings | 34.583 |
| describe_scenes | 11.590 |
| summarize_scenes | 8.999 |
| synthesize_synopsis | 15.436 |
| make_embedding | 4.180 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 49.050 |
| branch_yolo_total | 12.273 |
| branch_audio_total | 56.451 |
