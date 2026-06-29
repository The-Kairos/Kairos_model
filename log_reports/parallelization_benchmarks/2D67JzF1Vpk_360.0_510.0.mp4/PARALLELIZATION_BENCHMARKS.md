# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-21 09:34:14 UTC | 2D67JzF1Vpk_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 155.180 | 1.742 | 59.973 | 8.354 | 5.367 | 10.520 | 3.885 |
| 2026-06-21 21:01:35 UTC | 2D67JzF1Vpk_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 156.084 | 1.729 | 61.593 | 10.350 | 10.295 | 5.118 | 3.847 |

## 2026-06-21 09:34:14 UTC | 2D67JzF1Vpk_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/2D67JzF1Vpk_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `155.180` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.742 |
| save_clips | - |
| sample_frames | 3.195 |
| caption_frames | 44.639 |
| sample_fps | 6.625 |
| detect_object_yolo | 9.496 |
| audio_scan | 11.947 |
| asr_timings | 15.261 |
| ast_timings | 32.756 |
| describe_scenes | 8.354 |
| summarize_scenes | 5.367 |
| synthesize_synopsis | 10.520 |
| make_embedding | 3.885 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 47.839 |
| branch_yolo_total | 16.126 |
| branch_audio_total | 59.973 |

## 2026-06-21 21:01:35 UTC | 2D67JzF1Vpk_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/2D67JzF1Vpk_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `156.084` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.729 |
| save_clips | - |
| sample_frames | 3.300 |
| caption_frames | 42.214 |
| sample_fps | 6.602 |
| detect_object_yolo | 9.636 |
| audio_scan | 11.811 |
| asr_timings | 16.555 |
| ast_timings | 33.219 |
| describe_scenes | 10.350 |
| summarize_scenes | 10.295 |
| synthesize_synopsis | 5.118 |
| make_embedding | 3.847 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 45.520 |
| branch_yolo_total | 16.244 |
| branch_audio_total | 61.593 |
