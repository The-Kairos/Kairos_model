# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-23 15:48:56 UTC | 3G2WCQnH6Nk_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 169.698 | 0.773 | 58.223 | 18.285 | 17.145 | 22.048 | 3.358 |
| 2026-06-24 09:45:20 UTC | 3G2WCQnH6Nk_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 185.819 | 0.785 | 74.873 | 16.472 | 24.149 | 15.943 | 3.303 |

## 2026-06-23 15:48:56 UTC | 3G2WCQnH6Nk_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/3G2WCQnH6Nk_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `169.698` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.773 |
| save_clips | - |
| sample_frames | 1.119 |
| caption_frames | 36.543 |
| sample_fps | 2.215 |
| detect_object_yolo | 8.605 |
| audio_scan | 14.855 |
| asr_timings | 16.883 |
| ast_timings | 26.477 |
| describe_scenes | 18.285 |
| summarize_scenes | 17.145 |
| synthesize_synopsis | 22.048 |
| make_embedding | 3.358 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 37.668 |
| branch_yolo_total | 10.826 |
| branch_audio_total | 58.223 |

## 2026-06-24 09:45:20 UTC | 3G2WCQnH6Nk_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/3G2WCQnH6Nk_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `185.819` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.785 |
| save_clips | - |
| sample_frames | 1.119 |
| caption_frames | 36.785 |
| sample_fps | 2.259 |
| detect_object_yolo | 8.734 |
| audio_scan | 14.775 |
| asr_timings | 33.739 |
| ast_timings | 26.352 |
| describe_scenes | 16.472 |
| summarize_scenes | 24.149 |
| synthesize_synopsis | 15.943 |
| make_embedding | 3.303 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 37.909 |
| branch_yolo_total | 10.999 |
| branch_audio_total | 74.873 |
