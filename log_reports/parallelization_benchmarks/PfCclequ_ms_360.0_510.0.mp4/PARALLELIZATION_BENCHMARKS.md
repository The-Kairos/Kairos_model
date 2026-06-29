# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 14:27:04 UTC | PfCclequ_ms_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 169.233 | 0.780 | 47.491 | 23.479 | 15.787 | 29.016 | 3.366 |

## 2026-06-25 14:27:04 UTC | PfCclequ_ms_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/PfCclequ_ms_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `169.233` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.780 |
| save_clips | - |
| sample_frames | 0.782 |
| caption_frames | 36.349 |
| sample_fps | 2.152 |
| detect_object_yolo | 8.513 |
| audio_scan | 8.881 |
| asr_timings | 11.394 |
| ast_timings | 27.208 |
| describe_scenes | 23.479 |
| summarize_scenes | 15.787 |
| synthesize_synopsis | 29.016 |
| make_embedding | 3.366 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 37.137 |
| branch_yolo_total | 10.671 |
| branch_audio_total | 47.491 |
