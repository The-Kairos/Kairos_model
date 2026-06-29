# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-21 22:48:31 UTC | 4HeSJ7tMo48_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 204.146 | 0.699 | 106.037 | 12.611 | 10.197 | 4.243 | 4.750 |

## 2026-06-21 22:48:31 UTC | 4HeSJ7tMo48_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/4HeSJ7tMo48_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `204.146` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.699 |
| save_clips | - |
| sample_frames | 1.527 |
| caption_frames | 50.015 |
| sample_fps | 2.330 |
| detect_object_yolo | 10.332 |
| audio_scan | 14.977 |
| asr_timings | 52.940 |
| ast_timings | 38.112 |
| describe_scenes | 12.611 |
| summarize_scenes | 10.197 |
| synthesize_synopsis | 4.243 |
| make_embedding | 4.750 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 51.548 |
| branch_yolo_total | 12.668 |
| branch_audio_total | 106.037 |
