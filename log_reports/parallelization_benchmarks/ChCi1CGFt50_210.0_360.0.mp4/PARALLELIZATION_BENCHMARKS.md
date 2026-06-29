# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 20:48:48 UTC | ChCi1CGFt50_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 233.362 | 0.677 | 73.439 | 18.966 | 32.012 | 20.459 | 6.089 |

## 2026-06-24 20:48:48 UTC | ChCi1CGFt50_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/ChCi1CGFt50_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `233.362` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.677 |
| save_clips | - |
| sample_frames | 1.607 |
| caption_frames | 63.900 |
| sample_fps | 2.464 |
| detect_object_yolo | 12.280 |
| audio_scan | 15.120 |
| asr_timings | 8.885 |
| ast_timings | 49.426 |
| describe_scenes | 18.966 |
| summarize_scenes | 32.012 |
| synthesize_synopsis | 20.459 |
| make_embedding | 6.089 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 65.512 |
| branch_yolo_total | 14.750 |
| branch_audio_total | 73.439 |
