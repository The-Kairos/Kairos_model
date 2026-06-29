# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 00:53:46 UTC | b5DIp2wHa8Q_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 166.141 | 0.641 | 58.649 | 15.630 | 17.738 | 16.738 | 3.716 |

## 2026-06-26 00:53:46 UTC | b5DIp2wHa8Q_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/b5DIp2wHa8Q_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `166.141` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.641 |
| save_clips | - |
| sample_frames | 1.014 |
| caption_frames | 39.593 |
| sample_fps | 2.076 |
| detect_object_yolo | 8.952 |
| audio_scan | 15.914 |
| asr_timings | 12.750 |
| ast_timings | 29.977 |
| describe_scenes | 15.630 |
| summarize_scenes | 17.738 |
| synthesize_synopsis | 16.738 |
| make_embedding | 3.716 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 40.613 |
| branch_yolo_total | 11.034 |
| branch_audio_total | 58.649 |
