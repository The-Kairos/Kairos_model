# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-21 21:08:16 UTC | 2I9-kvemtSU_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 147.016 | 0.768 | 55.627 | 10.004 | 10.003 | 8.238 | 3.936 |

## 2026-06-21 21:08:16 UTC | 2I9-kvemtSU_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/2I9-kvemtSU_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `147.016` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.768 |
| save_clips | - |
| sample_frames | 1.173 |
| caption_frames | 43.822 |
| sample_fps | 2.290 |
| detect_object_yolo | 9.744 |
| audio_scan | 13.861 |
| asr_timings | 8.975 |
| ast_timings | 32.783 |
| describe_scenes | 10.004 |
| summarize_scenes | 10.003 |
| synthesize_synopsis | 8.238 |
| make_embedding | 3.936 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 45.001 |
| branch_yolo_total | 12.040 |
| branch_audio_total | 55.627 |
