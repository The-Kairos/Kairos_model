# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 08:34:56 UTC | iJRwe2oCX0E_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 210.893 | 0.819 | 58.600 | 16.894 | 32.471 | 36.909 | 4.223 |

## 2026-06-26 08:34:56 UTC | iJRwe2oCX0E_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/iJRwe2oCX0E_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `210.893` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.819 |
| save_clips | - |
| sample_frames | 1.273 |
| caption_frames | 46.407 |
| sample_fps | 2.354 |
| detect_object_yolo | 9.548 |
| audio_scan | 15.089 |
| asr_timings | 8.110 |
| ast_timings | 35.393 |
| describe_scenes | 16.894 |
| summarize_scenes | 32.471 |
| synthesize_synopsis | 36.909 |
| make_embedding | 4.223 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 47.685 |
| branch_yolo_total | 11.908 |
| branch_audio_total | 58.600 |
