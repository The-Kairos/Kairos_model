# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 05:24:48 UTC | ysd6xzuJ6S4_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 154.350 | 0.677 | 57.244 | 11.189 | 8.424 | 13.630 | 4.130 |

## 2026-06-27 05:24:48 UTC | ysd6xzuJ6S4_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/ysd6xzuJ6S4_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `154.350` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.677 |
| save_clips | - |
| sample_frames | 1.712 |
| caption_frames | 43.920 |
| sample_fps | 2.390 |
| detect_object_yolo | 9.624 |
| audio_scan | 14.004 |
| asr_timings | 7.370 |
| ast_timings | 35.862 |
| describe_scenes | 11.189 |
| summarize_scenes | 8.424 |
| synthesize_synopsis | 13.630 |
| make_embedding | 4.130 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 45.639 |
| branch_yolo_total | 12.019 |
| branch_audio_total | 57.244 |
