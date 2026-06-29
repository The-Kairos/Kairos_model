# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 06:56:18 UTC | KyjkOyUDXuY_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 177.462 | 0.785 | 71.250 | 11.521 | 12.862 | 18.560 | 3.691 |

## 2026-06-25 06:56:18 UTC | KyjkOyUDXuY_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/KyjkOyUDXuY_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `177.462` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.785 |
| save_clips | - |
| sample_frames | 0.989 |
| caption_frames | 44.516 |
| sample_fps | 2.214 |
| detect_object_yolo | 9.644 |
| audio_scan | 6.443 |
| asr_timings | 35.201 |
| ast_timings | 29.596 |
| describe_scenes | 11.521 |
| summarize_scenes | 12.862 |
| synthesize_synopsis | 18.560 |
| make_embedding | 3.691 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 45.510 |
| branch_yolo_total | 11.864 |
| branch_audio_total | 71.250 |
