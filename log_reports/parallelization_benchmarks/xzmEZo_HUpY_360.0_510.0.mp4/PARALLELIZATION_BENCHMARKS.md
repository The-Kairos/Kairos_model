# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 04:41:14 UTC | xzmEZo_HUpY_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 169.779 | 0.800 | 63.115 | 14.302 | 8.415 | 7.473 | 5.120 |

## 2026-06-27 04:41:14 UTC | xzmEZo_HUpY_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/xzmEZo_HUpY_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `169.779` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.800 |
| save_clips | - |
| sample_frames | 1.438 |
| caption_frames | 54.021 |
| sample_fps | 2.513 |
| detect_object_yolo | 11.162 |
| audio_scan | 7.660 |
| asr_timings | 14.852 |
| ast_timings | 40.594 |
| describe_scenes | 14.302 |
| summarize_scenes | 8.415 |
| synthesize_synopsis | 7.473 |
| make_embedding | 5.120 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 55.464 |
| branch_yolo_total | 13.681 |
| branch_audio_total | 63.115 |
