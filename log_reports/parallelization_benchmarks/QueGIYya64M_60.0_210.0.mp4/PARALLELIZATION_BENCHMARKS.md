# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 15:57:47 UTC | QueGIYya64M_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 240.592 | 0.788 | 100.830 | 22.476 | 29.992 | 14.552 | 4.821 |

## 2026-06-25 15:57:47 UTC | QueGIYya64M_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/QueGIYya64M_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `240.592` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.788 |
| save_clips | - |
| sample_frames | 1.479 |
| caption_frames | 51.220 |
| sample_fps | 2.484 |
| detect_object_yolo | 10.510 |
| audio_scan | 12.213 |
| asr_timings | 50.930 |
| ast_timings | 37.678 |
| describe_scenes | 22.476 |
| summarize_scenes | 29.992 |
| synthesize_synopsis | 14.552 |
| make_embedding | 4.821 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 52.705 |
| branch_yolo_total | 13.000 |
| branch_audio_total | 100.830 |
