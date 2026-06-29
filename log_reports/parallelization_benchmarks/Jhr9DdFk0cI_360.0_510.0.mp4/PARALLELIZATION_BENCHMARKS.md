# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 05:19:21 UTC | Jhr9DdFk0cI_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 187.603 | 0.782 | 94.857 | 14.846 | 9.403 | 11.473 | 3.550 |

## 2026-06-25 05:19:21 UTC | Jhr9DdFk0cI_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/Jhr9DdFk0cI_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `187.603` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.782 |
| save_clips | - |
| sample_frames | 0.900 |
| caption_frames | 39.928 |
| sample_fps | 2.216 |
| detect_object_yolo | 8.250 |
| audio_scan | 13.903 |
| asr_timings | 52.048 |
| ast_timings | 28.898 |
| describe_scenes | 14.846 |
| summarize_scenes | 9.403 |
| synthesize_synopsis | 11.473 |
| make_embedding | 3.550 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 40.834 |
| branch_yolo_total | 10.472 |
| branch_audio_total | 94.857 |
