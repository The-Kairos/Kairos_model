# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 01:07:37 UTC | uWgykeg3gug_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 159.427 | 0.802 | 60.249 | 12.068 | 18.124 | 11.888 | 3.539 |

## 2026-06-27 01:07:37 UTC | uWgykeg3gug_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/uWgykeg3gug_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `159.427` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.802 |
| save_clips | - |
| sample_frames | 1.315 |
| caption_frames | 38.948 |
| sample_fps | 2.354 |
| detect_object_yolo | 8.703 |
| audio_scan | 15.026 |
| asr_timings | 15.498 |
| ast_timings | 29.717 |
| describe_scenes | 12.068 |
| summarize_scenes | 18.124 |
| synthesize_synopsis | 11.888 |
| make_embedding | 3.539 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 40.269 |
| branch_yolo_total | 11.062 |
| branch_audio_total | 60.249 |
