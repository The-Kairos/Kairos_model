# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-21 09:36:08 UTC | 2D67JzF1Vpk_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 113.369 | 1.743 | 40.964 | 8.215 | 8.757 | 7.409 | 2.851 |
| 2026-06-21 21:03:27 UTC | 2D67JzF1Vpk_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 110.027 | 1.770 | 42.100 | 6.446 | 6.290 | 6.567 | 2.880 |

## 2026-06-21 09:36:08 UTC | 2D67JzF1Vpk_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/2D67JzF1Vpk_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `113.369` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.743 |
| save_clips | - |
| sample_frames | 2.069 |
| caption_frames | 27.138 |
| sample_fps | 5.648 |
| detect_object_yolo | 7.222 |
| audio_scan | 8.505 |
| asr_timings | 11.570 |
| ast_timings | 20.880 |
| describe_scenes | 8.215 |
| summarize_scenes | 8.757 |
| synthesize_synopsis | 7.409 |
| make_embedding | 2.851 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 29.213 |
| branch_yolo_total | 12.876 |
| branch_audio_total | 40.964 |

## 2026-06-21 21:03:27 UTC | 2D67JzF1Vpk_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/2D67JzF1Vpk_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `110.027` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.770 |
| save_clips | - |
| sample_frames | 2.010 |
| caption_frames | 27.318 |
| sample_fps | 5.807 |
| detect_object_yolo | 7.443 |
| audio_scan | 8.576 |
| asr_timings | 12.293 |
| ast_timings | 21.223 |
| describe_scenes | 6.446 |
| summarize_scenes | 6.290 |
| synthesize_synopsis | 6.567 |
| make_embedding | 2.880 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 29.334 |
| branch_yolo_total | 13.255 |
| branch_audio_total | 42.100 |
