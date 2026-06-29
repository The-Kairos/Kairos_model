# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 22:38:41 UTC | DOET406zX8A_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 80.594 | 0.671 | 22.832 | 8.383 | 11.198 | 8.840 | 2.301 |

## 2026-06-24 22:38:41 UTC | DOET406zX8A_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/DOET406zX8A_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `80.594` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.671 |
| save_clips | - |
| sample_frames | 0.476 |
| caption_frames | 22.350 |
| sample_fps | 1.791 |
| detect_object_yolo | 6.407 |
| audio_scan | 3.862 |
| asr_timings | 0.000 |
| ast_timings | 12.774 |
| describe_scenes | 8.383 |
| summarize_scenes | 11.198 |
| synthesize_synopsis | 8.840 |
| make_embedding | 2.301 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 22.832 |
| branch_yolo_total | 8.204 |
| branch_audio_total | 16.644 |
