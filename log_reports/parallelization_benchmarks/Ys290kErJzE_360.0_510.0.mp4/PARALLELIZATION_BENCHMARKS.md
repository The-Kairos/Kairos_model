# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 22:03:08 UTC | Ys290kErJzE_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 138.468 | 0.642 | 57.506 | 9.476 | 9.022 | 14.801 | 3.048 |

## 2026-06-25 22:03:08 UTC | Ys290kErJzE_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/Ys290kErJzE_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `138.468` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.642 |
| save_clips | - |
| sample_frames | 0.810 |
| caption_frames | 31.852 |
| sample_fps | 1.986 |
| detect_object_yolo | 7.923 |
| audio_scan | 7.566 |
| asr_timings | 26.076 |
| ast_timings | 23.856 |
| describe_scenes | 9.476 |
| summarize_scenes | 9.022 |
| synthesize_synopsis | 14.801 |
| make_embedding | 3.048 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 32.668 |
| branch_yolo_total | 9.914 |
| branch_audio_total | 57.506 |
