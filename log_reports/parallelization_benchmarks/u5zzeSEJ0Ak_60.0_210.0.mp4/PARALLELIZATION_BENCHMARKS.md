# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 00:29:51 UTC | u5zzeSEJ0Ak_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 204.857 | 0.661 | 71.977 | 18.234 | 10.877 | 8.663 | 6.459 |

## 2026-06-27 00:29:51 UTC | u5zzeSEJ0Ak_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/u5zzeSEJ0Ak_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `204.857` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.661 |
| save_clips | - |
| sample_frames | 1.895 |
| caption_frames | 70.076 |
| sample_fps | 2.527 |
| detect_object_yolo | 13.041 |
| audio_scan | 7.578 |
| asr_timings | 11.040 |
| ast_timings | 52.342 |
| describe_scenes | 18.234 |
| summarize_scenes | 10.877 |
| synthesize_synopsis | 8.663 |
| make_embedding | 6.459 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 71.977 |
| branch_yolo_total | 15.574 |
| branch_audio_total | 70.969 |
