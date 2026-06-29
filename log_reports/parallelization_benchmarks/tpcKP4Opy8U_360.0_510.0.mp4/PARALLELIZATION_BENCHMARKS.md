# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 23:57:30 UTC | tpcKP4Opy8U_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 157.339 | 0.666 | 54.664 | 14.067 | 8.450 | 9.385 | 4.473 |

## 2026-06-26 23:57:30 UTC | tpcKP4Opy8U_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/tpcKP4Opy8U_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `157.339` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.666 |
| save_clips | - |
| sample_frames | 1.881 |
| caption_frames | 49.843 |
| sample_fps | 2.377 |
| detect_object_yolo | 10.089 |
| audio_scan | 6.475 |
| asr_timings | 8.813 |
| ast_timings | 39.367 |
| describe_scenes | 14.067 |
| summarize_scenes | 8.450 |
| synthesize_synopsis | 9.385 |
| make_embedding | 4.473 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 51.731 |
| branch_yolo_total | 12.473 |
| branch_audio_total | 54.664 |
