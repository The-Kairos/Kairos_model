# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 22:26:00 UTC | ZR7Z74UY2TY_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 133.209 | 0.662 | 49.452 | 11.747 | 10.685 | 10.083 | 3.304 |

## 2026-06-25 22:26:00 UTC | ZR7Z74UY2TY_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/ZR7Z74UY2TY_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `133.209` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.662 |
| save_clips | - |
| sample_frames | 0.991 |
| caption_frames | 34.458 |
| sample_fps | 2.084 |
| detect_object_yolo | 8.342 |
| audio_scan | 13.814 |
| asr_timings | 8.182 |
| ast_timings | 27.448 |
| describe_scenes | 11.747 |
| summarize_scenes | 10.685 |
| synthesize_synopsis | 10.083 |
| make_embedding | 3.304 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 35.455 |
| branch_yolo_total | 10.432 |
| branch_audio_total | 49.452 |
