# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 17:42:23 UTC | lyGaTk4MLVM_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 250.608 | 0.762 | 74.481 | 28.626 | 39.020 | 19.436 | 6.127 |

## 2026-06-26 17:42:23 UTC | lyGaTk4MLVM_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/lyGaTk4MLVM_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `250.608` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.762 |
| save_clips | - |
| sample_frames | 1.807 |
| caption_frames | 64.332 |
| sample_fps | 2.583 |
| detect_object_yolo | 12.012 |
| audio_scan | 15.059 |
| asr_timings | 9.982 |
| ast_timings | 49.431 |
| describe_scenes | 28.626 |
| summarize_scenes | 39.020 |
| synthesize_synopsis | 19.436 |
| make_embedding | 6.127 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 66.144 |
| branch_yolo_total | 14.600 |
| branch_audio_total | 74.481 |
