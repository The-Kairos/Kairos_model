# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 04:19:36 UTC | fMStD2X1434_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 150.715 | 0.837 | 68.448 | 10.077 | 9.403 | 7.945 | 3.377 |

## 2026-06-26 04:19:36 UTC | fMStD2X1434_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/fMStD2X1434_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `150.715` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.837 |
| save_clips | - |
| sample_frames | 1.114 |
| caption_frames | 36.808 |
| sample_fps | 2.307 |
| detect_object_yolo | 8.972 |
| audio_scan | 8.646 |
| asr_timings | 32.192 |
| ast_timings | 27.601 |
| describe_scenes | 10.077 |
| summarize_scenes | 9.403 |
| synthesize_synopsis | 7.945 |
| make_embedding | 3.377 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 37.928 |
| branch_yolo_total | 11.285 |
| branch_audio_total | 68.448 |
