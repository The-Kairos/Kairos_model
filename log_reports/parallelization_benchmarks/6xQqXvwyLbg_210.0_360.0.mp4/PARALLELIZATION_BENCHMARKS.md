# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 13:15:58 UTC | 6xQqXvwyLbg_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 229.944 | 0.834 | 87.682 | 24.362 | 37.375 | 18.308 | 3.989 |

## 2026-06-24 13:15:58 UTC | 6xQqXvwyLbg_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/6xQqXvwyLbg_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `229.944` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.834 |
| save_clips | - |
| sample_frames | 1.390 |
| caption_frames | 43.156 |
| sample_fps | 2.343 |
| detect_object_yolo | 9.111 |
| audio_scan | 13.907 |
| asr_timings | 41.059 |
| ast_timings | 32.708 |
| describe_scenes | 24.362 |
| summarize_scenes | 37.375 |
| synthesize_synopsis | 18.308 |
| make_embedding | 3.989 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 44.552 |
| branch_yolo_total | 11.459 |
| branch_audio_total | 87.682 |
