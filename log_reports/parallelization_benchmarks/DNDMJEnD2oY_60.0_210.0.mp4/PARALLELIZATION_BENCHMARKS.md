# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 22:33:42 UTC | DNDMJEnD2oY_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 188.362 | 0.701 | 64.878 | 15.750 | 19.920 | 9.424 | 5.097 |

## 2026-06-24 22:33:42 UTC | DNDMJEnD2oY_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/DNDMJEnD2oY_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `188.362` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.701 |
| save_clips | - |
| sample_frames | 1.462 |
| caption_frames | 56.425 |
| sample_fps | 2.397 |
| detect_object_yolo | 10.867 |
| audio_scan | 13.922 |
| asr_timings | 9.094 |
| ast_timings | 41.854 |
| describe_scenes | 15.750 |
| summarize_scenes | 19.920 |
| synthesize_synopsis | 9.424 |
| make_embedding | 5.097 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 57.893 |
| branch_yolo_total | 13.270 |
| branch_audio_total | 64.878 |
