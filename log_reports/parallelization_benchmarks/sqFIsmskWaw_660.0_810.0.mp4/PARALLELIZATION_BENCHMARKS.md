# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 22:43:29 UTC | sqFIsmskWaw_660.0_810.0.mp4 | sequential | gemini | gemini-embedding-001 | 166.365 | 0.780 | 60.576 | 14.306 | 9.378 | 14.974 | 4.069 |

## 2026-06-26 22:43:29 UTC | sqFIsmskWaw_660.0_810.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/sqFIsmskWaw_660.0_810.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `166.365` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.780 |
| save_clips | - |
| sample_frames | 1.570 |
| caption_frames | 46.548 |
| sample_fps | 2.436 |
| detect_object_yolo | 10.286 |
| audio_scan | 15.988 |
| asr_timings | 8.953 |
| ast_timings | 35.627 |
| describe_scenes | 14.306 |
| summarize_scenes | 9.378 |
| synthesize_synopsis | 14.974 |
| make_embedding | 4.069 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 48.124 |
| branch_yolo_total | 12.728 |
| branch_audio_total | 60.576 |
