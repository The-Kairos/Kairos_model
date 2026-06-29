# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 23:10:21 UTC | _6hnl_BrFvs_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 125.537 | 0.962 | 40.099 | 16.410 | 6.873 | 11.136 | 3.574 |

## 2026-06-25 23:10:21 UTC | _6hnl_BrFvs_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/_6hnl_BrFvs_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `125.537` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.962 |
| save_clips | - |
| sample_frames | 1.665 |
| caption_frames | 38.428 |
| sample_fps | 2.524 |
| detect_object_yolo | 8.054 |
| audio_scan | 3.859 |
| asr_timings | 0.000 |
| ast_timings | 30.627 |
| describe_scenes | 16.410 |
| summarize_scenes | 6.873 |
| synthesize_synopsis | 11.136 |
| make_embedding | 3.574 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 40.099 |
| branch_yolo_total | 10.583 |
| branch_audio_total | 34.494 |
