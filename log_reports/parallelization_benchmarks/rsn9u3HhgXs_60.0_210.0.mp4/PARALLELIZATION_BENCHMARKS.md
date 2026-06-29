# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 19:11:18 UTC | rsn9u3HhgXs_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 184.131 | 0.799 | 77.261 | 15.707 | 9.381 | 17.604 | 3.933 |

## 2026-06-26 19:11:18 UTC | rsn9u3HhgXs_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/rsn9u3HhgXs_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `184.131` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.799 |
| save_clips | - |
| sample_frames | 1.379 |
| caption_frames | 44.624 |
| sample_fps | 2.401 |
| detect_object_yolo | 9.619 |
| audio_scan | 10.700 |
| asr_timings | 33.707 |
| ast_timings | 32.846 |
| describe_scenes | 15.707 |
| summarize_scenes | 9.381 |
| synthesize_synopsis | 17.604 |
| make_embedding | 3.933 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 46.009 |
| branch_yolo_total | 12.026 |
| branch_audio_total | 77.261 |
