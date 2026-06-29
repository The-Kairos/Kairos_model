# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 01:22:38 UTC | GDi-Pbip33M_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 192.575 | 0.870 | 65.222 | 24.639 | 16.648 | 14.871 | 4.518 |

## 2026-06-25 01:22:38 UTC | GDi-Pbip33M_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/GDi-Pbip33M_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `192.575` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.870 |
| save_clips | - |
| sample_frames | 1.618 |
| caption_frames | 49.849 |
| sample_fps | 2.539 |
| detect_object_yolo | 10.376 |
| audio_scan | 14.862 |
| asr_timings | 12.354 |
| ast_timings | 37.998 |
| describe_scenes | 24.639 |
| summarize_scenes | 16.648 |
| synthesize_synopsis | 14.871 |
| make_embedding | 4.518 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 51.473 |
| branch_yolo_total | 12.920 |
| branch_audio_total | 65.222 |
