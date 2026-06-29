# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-23 16:26:38 UTC | 3cS4WYK4G7U_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 129.546 | 0.763 | 40.509 | 10.856 | 8.742 | 33.560 | 2.293 |
| 2026-06-24 10:22:12 UTC | 3cS4WYK4G7U_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 124.459 | 0.755 | 40.666 | 13.707 | 8.590 | 25.714 | 2.286 |

## 2026-06-23 16:26:38 UTC | 3cS4WYK4G7U_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/3cS4WYK4G7U_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `129.546` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.763 |
| save_clips | - |
| sample_frames | 0.521 |
| caption_frames | 21.933 |
| sample_fps | 1.929 |
| detect_object_yolo | 7.035 |
| audio_scan | 15.789 |
| asr_timings | 9.787 |
| ast_timings | 14.924 |
| describe_scenes | 10.856 |
| summarize_scenes | 8.742 |
| synthesize_synopsis | 33.560 |
| make_embedding | 2.293 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 22.460 |
| branch_yolo_total | 8.970 |
| branch_audio_total | 40.509 |

## 2026-06-24 10:22:12 UTC | 3cS4WYK4G7U_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/3cS4WYK4G7U_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `124.459` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.755 |
| save_clips | - |
| sample_frames | 0.514 |
| caption_frames | 21.654 |
| sample_fps | 1.953 |
| detect_object_yolo | 7.254 |
| audio_scan | 16.019 |
| asr_timings | 9.830 |
| ast_timings | 14.808 |
| describe_scenes | 13.707 |
| summarize_scenes | 8.590 |
| synthesize_synopsis | 25.714 |
| make_embedding | 2.286 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 22.174 |
| branch_yolo_total | 9.213 |
| branch_audio_total | 40.666 |
