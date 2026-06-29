# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 12:51:21 UTC | -eQfg7HJuKc_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 128.234 | 0.770 | 49.426 | 12.630 | 6.311 | 5.605 | 3.334 |

## 2026-06-27 12:51:21 UTC | -eQfg7HJuKc_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/-eQfg7HJuKc_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `128.234` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.770 |
| save_clips | - |
| sample_frames | 0.967 |
| caption_frames | 36.969 |
| sample_fps | 2.208 |
| detect_object_yolo | 8.592 |
| audio_scan | 14.913 |
| asr_timings | 8.190 |
| ast_timings | 26.315 |
| describe_scenes | 12.630 |
| summarize_scenes | 6.311 |
| synthesize_synopsis | 5.605 |
| make_embedding | 3.334 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 37.942 |
| branch_yolo_total | 10.807 |
| branch_audio_total | 49.426 |
