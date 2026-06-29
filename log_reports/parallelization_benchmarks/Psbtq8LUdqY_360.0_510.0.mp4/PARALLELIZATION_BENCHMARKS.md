# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 14:37:36 UTC | Psbtq8LUdqY_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 236.750 | 0.794 | 69.361 | 24.275 | 35.634 | 17.502 | 5.835 |

## 2026-06-25 14:37:36 UTC | Psbtq8LUdqY_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/Psbtq8LUdqY_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `236.750` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.794 |
| save_clips | - |
| sample_frames | 1.508 |
| caption_frames | 65.455 |
| sample_fps | 2.605 |
| detect_object_yolo | 12.329 |
| audio_scan | 14.367 |
| asr_timings | 9.294 |
| ast_timings | 45.691 |
| describe_scenes | 24.275 |
| summarize_scenes | 35.634 |
| synthesize_synopsis | 17.502 |
| make_embedding | 5.835 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 66.969 |
| branch_yolo_total | 14.940 |
| branch_audio_total | 69.361 |
