# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 03:21:01 UTC | HJiDesesiWg_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 212.059 | 0.677 | 67.551 | 21.681 | 24.329 | 16.979 | 5.334 |

## 2026-06-25 03:21:01 UTC | HJiDesesiWg_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/HJiDesesiWg_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `212.059` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.677 |
| save_clips | - |
| sample_frames | 1.689 |
| caption_frames | 58.614 |
| sample_fps | 2.466 |
| detect_object_yolo | 11.337 |
| audio_scan | 14.903 |
| asr_timings | 8.877 |
| ast_timings | 43.763 |
| describe_scenes | 21.681 |
| summarize_scenes | 24.329 |
| synthesize_synopsis | 16.979 |
| make_embedding | 5.334 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 60.309 |
| branch_yolo_total | 13.809 |
| branch_audio_total | 67.551 |
