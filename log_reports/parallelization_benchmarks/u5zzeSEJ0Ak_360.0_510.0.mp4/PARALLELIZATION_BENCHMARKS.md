# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 00:26:25 UTC | u5zzeSEJ0Ak_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 168.311 | 0.699 | 56.758 | 15.295 | 9.688 | 12.760 | 4.700 |

## 2026-06-27 00:26:25 UTC | u5zzeSEJ0Ak_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/u5zzeSEJ0Ak_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `168.311` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.699 |
| save_clips | - |
| sample_frames | 1.461 |
| caption_frames | 52.321 |
| sample_fps | 2.352 |
| detect_object_yolo | 10.839 |
| audio_scan | 6.505 |
| asr_timings | 10.901 |
| ast_timings | 39.343 |
| describe_scenes | 15.295 |
| summarize_scenes | 9.688 |
| synthesize_synopsis | 12.760 |
| make_embedding | 4.700 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 53.789 |
| branch_yolo_total | 13.197 |
| branch_audio_total | 56.758 |
