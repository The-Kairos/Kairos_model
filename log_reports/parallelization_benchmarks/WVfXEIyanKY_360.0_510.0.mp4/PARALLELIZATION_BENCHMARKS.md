# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 20:29:27 UTC | WVfXEIyanKY_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 132.563 | 0.776 | 49.991 | 10.411 | 6.677 | 13.843 | 3.269 |

## 2026-06-25 20:29:27 UTC | WVfXEIyanKY_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/WVfXEIyanKY_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `132.563` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.776 |
| save_clips | - |
| sample_frames | 0.823 |
| caption_frames | 34.962 |
| sample_fps | 2.127 |
| detect_object_yolo | 8.247 |
| audio_scan | 12.916 |
| asr_timings | 9.683 |
| ast_timings | 27.383 |
| describe_scenes | 10.411 |
| summarize_scenes | 6.677 |
| synthesize_synopsis | 13.843 |
| make_embedding | 3.269 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 35.791 |
| branch_yolo_total | 10.379 |
| branch_audio_total | 49.991 |
