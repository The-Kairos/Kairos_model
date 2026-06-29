# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 04:36:47 UTC | fiyIhcNuSaA_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 163.170 | 0.876 | 97.486 | 6.582 | 5.938 | 15.458 | 2.312 |

## 2026-06-26 04:36:47 UTC | fiyIhcNuSaA_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/fiyIhcNuSaA_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `163.170` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.876 |
| save_clips | - |
| sample_frames | 0.586 |
| caption_frames | 22.958 |
| sample_fps | 2.050 |
| detect_object_yolo | 7.515 |
| audio_scan | 16.281 |
| asr_timings | 65.253 |
| ast_timings | 15.942 |
| describe_scenes | 6.582 |
| summarize_scenes | 5.938 |
| synthesize_synopsis | 15.458 |
| make_embedding | 2.312 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 23.550 |
| branch_yolo_total | 9.571 |
| branch_audio_total | 97.486 |
