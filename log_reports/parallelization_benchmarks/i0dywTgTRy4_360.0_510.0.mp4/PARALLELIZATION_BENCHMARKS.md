# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 07:45:20 UTC | i0dywTgTRy4_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 125.493 | 0.670 | 41.782 | 10.615 | 6.828 | 27.151 | 2.521 |

## 2026-06-26 07:45:20 UTC | i0dywTgTRy4_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/i0dywTgTRy4_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `125.493` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.670 |
| save_clips | - |
| sample_frames | 0.812 |
| caption_frames | 24.474 |
| sample_fps | 1.961 |
| detect_object_yolo | 7.246 |
| audio_scan | 14.137 |
| asr_timings | 9.435 |
| ast_timings | 18.201 |
| describe_scenes | 10.615 |
| summarize_scenes | 6.828 |
| synthesize_synopsis | 27.151 |
| make_embedding | 2.521 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 25.293 |
| branch_yolo_total | 9.213 |
| branch_audio_total | 41.782 |
