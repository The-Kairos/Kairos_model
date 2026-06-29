# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 06:22:42 UTC | hYMq3e9c8lY_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 191.005 | 0.580 | 61.294 | 16.367 | 13.704 | 24.568 | 4.666 |

## 2026-06-26 06:22:42 UTC | hYMq3e9c8lY_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/hYMq3e9c8lY_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `191.005` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.580 |
| save_clips | - |
| sample_frames | 1.286 |
| caption_frames | 54.775 |
| sample_fps | 2.085 |
| detect_object_yolo | 10.245 |
| audio_scan | 13.692 |
| asr_timings | 9.289 |
| ast_timings | 38.305 |
| describe_scenes | 16.367 |
| summarize_scenes | 13.704 |
| synthesize_synopsis | 24.568 |
| make_embedding | 4.666 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 56.068 |
| branch_yolo_total | 12.336 |
| branch_audio_total | 61.294 |
