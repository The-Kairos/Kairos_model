# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 13:11:08 UTC | -ruy-w0bxvA_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 113.480 | 0.744 | 46.527 | 6.014 | 5.597 | 6.920 | 2.992 |

## 2026-06-27 13:11:08 UTC | -ruy-w0bxvA_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/-ruy-w0bxvA_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `113.480` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.744 |
| save_clips | - |
| sample_frames | 0.760 |
| caption_frames | 32.860 |
| sample_fps | 2.044 |
| detect_object_yolo | 7.622 |
| audio_scan | 13.845 |
| asr_timings | 9.332 |
| ast_timings | 23.342 |
| describe_scenes | 6.014 |
| summarize_scenes | 5.597 |
| synthesize_synopsis | 6.920 |
| make_embedding | 2.992 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 33.626 |
| branch_yolo_total | 9.671 |
| branch_audio_total | 46.527 |
