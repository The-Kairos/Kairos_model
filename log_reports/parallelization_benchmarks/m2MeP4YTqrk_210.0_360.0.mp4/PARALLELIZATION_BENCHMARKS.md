# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 17:47:39 UTC | m2MeP4YTqrk_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 205.393 | 0.821 | 58.691 | 23.244 | 38.758 | 18.591 | 4.198 |

## 2026-06-26 17:47:39 UTC | m2MeP4YTqrk_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/m2MeP4YTqrk_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `205.393` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.821 |
| save_clips | - |
| sample_frames | 1.644 |
| caption_frames | 45.720 |
| sample_fps | 2.519 |
| detect_object_yolo | 9.789 |
| audio_scan | 11.853 |
| asr_timings | 12.595 |
| ast_timings | 34.234 |
| describe_scenes | 23.244 |
| summarize_scenes | 38.758 |
| synthesize_synopsis | 18.591 |
| make_embedding | 4.198 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 47.370 |
| branch_yolo_total | 12.314 |
| branch_audio_total | 58.691 |
