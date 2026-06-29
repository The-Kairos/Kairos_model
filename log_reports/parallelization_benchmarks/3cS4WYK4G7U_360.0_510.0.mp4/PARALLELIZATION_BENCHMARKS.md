# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-23 16:24:28 UTC | 3cS4WYK4G7U_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 91.178 | 0.757 | 26.062 | 9.685 | 6.088 | 30.544 | 1.255 |
| 2026-06-24 10:20:06 UTC | 3cS4WYK4G7U_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 107.133 | 0.773 | 26.338 | 6.841 | 4.569 | 50.505 | 1.383 |

## 2026-06-23 16:24:28 UTC | 3cS4WYK4G7U_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/3cS4WYK4G7U_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `91.178` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.757 |
| save_clips | - |
| sample_frames | 0.112 |
| caption_frames | 7.975 |
| sample_fps | 1.738 |
| detect_object_yolo | 5.607 |
| audio_scan | 11.645 |
| asr_timings | 10.145 |
| ast_timings | 4.263 |
| describe_scenes | 9.685 |
| summarize_scenes | 6.088 |
| synthesize_synopsis | 30.544 |
| make_embedding | 1.255 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 8.092 |
| branch_yolo_total | 7.351 |
| branch_audio_total | 26.062 |

## 2026-06-24 10:20:06 UTC | 3cS4WYK4G7U_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/3cS4WYK4G7U_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `107.133` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.773 |
| save_clips | - |
| sample_frames | 0.116 |
| caption_frames | 7.894 |
| sample_fps | 1.755 |
| detect_object_yolo | 5.584 |
| audio_scan | 11.715 |
| asr_timings | 10.245 |
| ast_timings | 4.369 |
| describe_scenes | 6.841 |
| summarize_scenes | 4.569 |
| synthesize_synopsis | 50.505 |
| make_embedding | 1.383 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 8.016 |
| branch_yolo_total | 7.345 |
| branch_audio_total | 26.338 |
