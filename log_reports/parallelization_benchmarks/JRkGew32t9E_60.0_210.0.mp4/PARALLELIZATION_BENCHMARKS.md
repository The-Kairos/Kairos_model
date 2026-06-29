# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 05:13:25 UTC | JRkGew32t9E_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 169.419 | 0.663 | 58.472 | 16.834 | 9.490 | 22.371 | 3.968 |

## 2026-06-25 05:13:25 UTC | JRkGew32t9E_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/JRkGew32t9E_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `169.419` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.663 |
| save_clips | - |
| sample_frames | 1.154 |
| caption_frames | 43.261 |
| sample_fps | 2.200 |
| detect_object_yolo | 9.574 |
| audio_scan | 16.124 |
| asr_timings | 10.004 |
| ast_timings | 32.337 |
| describe_scenes | 16.834 |
| summarize_scenes | 9.490 |
| synthesize_synopsis | 22.371 |
| make_embedding | 3.968 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 44.421 |
| branch_yolo_total | 11.779 |
| branch_audio_total | 58.472 |
