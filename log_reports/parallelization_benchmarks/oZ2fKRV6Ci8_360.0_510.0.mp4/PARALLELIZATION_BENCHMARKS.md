# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 17:17:33 UTC | oZ2fKRV6Ci8_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 189.140 | 0.766 | 70.622 | 14.265 | 10.352 | 10.060 | 5.407 |

## 2026-06-27 17:17:33 UTC | oZ2fKRV6Ci8_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/oZ2fKRV6Ci8_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `189.140` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.766 |
| save_clips | - |
| sample_frames | 1.234 |
| caption_frames | 61.065 |
| sample_fps | 2.482 |
| detect_object_yolo | 11.418 |
| audio_scan | 15.085 |
| asr_timings | 11.159 |
| ast_timings | 44.370 |
| describe_scenes | 14.265 |
| summarize_scenes | 10.352 |
| synthesize_synopsis | 10.060 |
| make_embedding | 5.407 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 62.305 |
| branch_yolo_total | 13.906 |
| branch_audio_total | 70.622 |
