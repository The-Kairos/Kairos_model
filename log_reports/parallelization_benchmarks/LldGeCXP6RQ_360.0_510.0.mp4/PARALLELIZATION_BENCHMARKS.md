# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 07:28:14 UTC | LldGeCXP6RQ_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 200.532 | 0.794 | 59.324 | 23.893 | 27.378 | 27.858 | 3.856 |

## 2026-06-25 07:28:14 UTC | LldGeCXP6RQ_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/LldGeCXP6RQ_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `200.532` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.794 |
| save_clips | - |
| sample_frames | 1.134 |
| caption_frames | 43.334 |
| sample_fps | 2.321 |
| detect_object_yolo | 9.226 |
| audio_scan | 14.871 |
| asr_timings | 12.538 |
| ast_timings | 31.907 |
| describe_scenes | 23.893 |
| summarize_scenes | 27.378 |
| synthesize_synopsis | 27.858 |
| make_embedding | 3.856 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 44.474 |
| branch_yolo_total | 11.552 |
| branch_audio_total | 59.324 |
