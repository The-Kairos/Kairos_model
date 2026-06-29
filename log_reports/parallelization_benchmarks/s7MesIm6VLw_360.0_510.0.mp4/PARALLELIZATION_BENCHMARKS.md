# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 19:30:16 UTC | s7MesIm6VLw_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 191.476 | 0.702 | 82.630 | 21.036 | 11.736 | 11.979 | 3.878 |

## 2026-06-26 19:30:16 UTC | s7MesIm6VLw_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/s7MesIm6VLw_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `191.476` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.702 |
| save_clips | - |
| sample_frames | 1.225 |
| caption_frames | 44.625 |
| sample_fps | 2.181 |
| detect_object_yolo | 9.989 |
| audio_scan | 14.328 |
| asr_timings | 34.875 |
| ast_timings | 33.418 |
| describe_scenes | 21.036 |
| summarize_scenes | 11.736 |
| synthesize_synopsis | 11.979 |
| make_embedding | 3.878 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 45.856 |
| branch_yolo_total | 12.176 |
| branch_audio_total | 82.630 |
