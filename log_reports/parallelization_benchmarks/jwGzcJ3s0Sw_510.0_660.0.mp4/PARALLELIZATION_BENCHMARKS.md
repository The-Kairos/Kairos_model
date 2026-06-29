# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 12:37:42 UTC | jwGzcJ3s0Sw_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 163.853 | 0.811 | 50.455 | 22.664 | 13.979 | 22.498 | 3.335 |

## 2026-06-26 12:37:42 UTC | jwGzcJ3s0Sw_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/jwGzcJ3s0Sw_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `163.853` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.811 |
| save_clips | - |
| sample_frames | 0.856 |
| caption_frames | 36.339 |
| sample_fps | 2.173 |
| detect_object_yolo | 9.299 |
| audio_scan | 9.752 |
| asr_timings | 12.676 |
| ast_timings | 28.018 |
| describe_scenes | 22.664 |
| summarize_scenes | 13.979 |
| synthesize_synopsis | 22.498 |
| make_embedding | 3.335 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 37.202 |
| branch_yolo_total | 11.477 |
| branch_audio_total | 50.455 |
