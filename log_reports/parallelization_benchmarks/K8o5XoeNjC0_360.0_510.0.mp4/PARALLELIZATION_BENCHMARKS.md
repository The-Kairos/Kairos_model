# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 06:01:53 UTC | K8o5XoeNjC0_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 176.580 | 0.654 | 51.180 | 22.295 | 27.886 | 15.755 | 3.567 |

## 2026-06-25 06:01:53 UTC | K8o5XoeNjC0_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/K8o5XoeNjC0_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `176.580` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.654 |
| save_clips | - |
| sample_frames | 1.035 |
| caption_frames | 41.696 |
| sample_fps | 2.115 |
| detect_object_yolo | 9.012 |
| audio_scan | 12.690 |
| asr_timings | 8.801 |
| ast_timings | 29.681 |
| describe_scenes | 22.295 |
| summarize_scenes | 27.886 |
| synthesize_synopsis | 15.755 |
| make_embedding | 3.567 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 42.737 |
| branch_yolo_total | 11.132 |
| branch_audio_total | 51.180 |
