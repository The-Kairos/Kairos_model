# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 16:29:35 UTC | nCsbm8Ze8oM_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 123.995 | 0.847 | 44.146 | 12.685 | 7.162 | 7.895 | 3.334 |

## 2026-06-27 16:29:35 UTC | nCsbm8Ze8oM_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/nCsbm8Ze8oM_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `123.995` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.847 |
| save_clips | - |
| sample_frames | 0.806 |
| caption_frames | 35.445 |
| sample_fps | 2.186 |
| detect_object_yolo | 8.100 |
| audio_scan | 5.334 |
| asr_timings | 11.588 |
| ast_timings | 27.215 |
| describe_scenes | 12.685 |
| summarize_scenes | 7.162 |
| synthesize_synopsis | 7.895 |
| make_embedding | 3.334 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 36.258 |
| branch_yolo_total | 10.292 |
| branch_audio_total | 44.146 |
