# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 00:27:50 UTC | FlONE32ZwmQ_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 129.430 | 0.627 | 47.843 | 9.268 | 9.144 | 13.068 | 3.089 |

## 2026-06-25 00:27:50 UTC | FlONE32ZwmQ_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/FlONE32ZwmQ_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `129.430` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.627 |
| save_clips | - |
| sample_frames | 0.892 |
| caption_frames | 33.536 |
| sample_fps | 1.979 |
| detect_object_yolo | 8.516 |
| audio_scan | 11.953 |
| asr_timings | 11.125 |
| ast_timings | 24.756 |
| describe_scenes | 9.268 |
| summarize_scenes | 9.144 |
| synthesize_synopsis | 13.068 |
| make_embedding | 3.089 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 34.434 |
| branch_yolo_total | 10.500 |
| branch_audio_total | 47.843 |
