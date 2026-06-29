# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 23:49:28 UTC | tmHVrxj0_Iw_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 169.857 | 0.852 | 74.648 | 13.428 | 9.273 | 7.694 | 4.384 |

## 2026-06-26 23:49:28 UTC | tmHVrxj0_Iw_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/tmHVrxj0_Iw_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `169.857` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.852 |
| save_clips | - |
| sample_frames | 1.348 |
| caption_frames | 44.649 |
| sample_fps | 2.293 |
| detect_object_yolo | 9.831 |
| audio_scan | 15.880 |
| asr_timings | 25.424 |
| ast_timings | 33.336 |
| describe_scenes | 13.428 |
| summarize_scenes | 9.273 |
| synthesize_synopsis | 7.694 |
| make_embedding | 4.384 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 46.002 |
| branch_yolo_total | 12.130 |
| branch_audio_total | 74.648 |
