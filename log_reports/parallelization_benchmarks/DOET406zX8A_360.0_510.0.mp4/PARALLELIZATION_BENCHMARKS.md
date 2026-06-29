# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 22:37:20 UTC | DOET406zX8A_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 135.206 | 0.668 | 48.768 | 12.808 | 7.663 | 8.981 | 4.152 |

## 2026-06-24 22:37:20 UTC | DOET406zX8A_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/DOET406zX8A_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `135.206` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.668 |
| save_clips | - |
| sample_frames | 1.393 |
| caption_frames | 47.369 |
| sample_fps | 2.265 |
| detect_object_yolo | 9.583 |
| audio_scan | 3.920 |
| asr_timings | 0.000 |
| ast_timings | 34.972 |
| describe_scenes | 12.808 |
| summarize_scenes | 7.663 |
| synthesize_synopsis | 8.981 |
| make_embedding | 4.152 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 48.768 |
| branch_yolo_total | 11.854 |
| branch_audio_total | 38.900 |
