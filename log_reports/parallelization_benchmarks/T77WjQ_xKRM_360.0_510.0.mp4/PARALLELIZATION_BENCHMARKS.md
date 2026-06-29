# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 17:42:53 UTC | T77WjQ_xKRM_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 92.897 | 0.665 | 33.794 | 6.421 | 7.487 | 9.261 | 2.322 |

## 2026-06-25 17:42:53 UTC | T77WjQ_xKRM_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/T77WjQ_xKRM_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `92.897` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.665 |
| save_clips | - |
| sample_frames | 0.567 |
| caption_frames | 22.077 |
| sample_fps | 1.821 |
| detect_object_yolo | 7.091 |
| audio_scan | 10.598 |
| asr_timings | 7.527 |
| ast_timings | 15.660 |
| describe_scenes | 6.421 |
| summarize_scenes | 7.487 |
| synthesize_synopsis | 9.261 |
| make_embedding | 2.322 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 22.650 |
| branch_yolo_total | 8.918 |
| branch_audio_total | 33.794 |
