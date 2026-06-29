# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-23 13:55:33 UTC | 1icyCzbxmmg_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 165.831 | 0.640 | 42.522 | 19.169 | 29.874 | 34.550 | 2.643 |
| 2026-06-27 15:21:54 UTC | 1icyCzbxmmg_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 105.210 | 0.662 | 42.893 | 8.572 | 6.198 | 9.985 | 2.537 |

## 2026-06-23 13:55:33 UTC | 1icyCzbxmmg_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/1icyCzbxmmg_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `165.831` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.640 |
| save_clips | - |
| sample_frames | 0.621 |
| caption_frames | 25.787 |
| sample_fps | 1.840 |
| detect_object_yolo | 6.791 |
| audio_scan | 15.885 |
| asr_timings | 8.524 |
| ast_timings | 18.105 |
| describe_scenes | 19.169 |
| summarize_scenes | 29.874 |
| synthesize_synopsis | 34.550 |
| make_embedding | 2.643 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 26.415 |
| branch_yolo_total | 8.637 |
| branch_audio_total | 42.522 |

## 2026-06-27 15:21:54 UTC | 1icyCzbxmmg_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/1icyCzbxmmg_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `105.210` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.662 |
| save_clips | - |
| sample_frames | 0.610 |
| caption_frames | 23.652 |
| sample_fps | 1.845 |
| detect_object_yolo | 6.841 |
| audio_scan | 15.955 |
| asr_timings | 8.543 |
| ast_timings | 18.386 |
| describe_scenes | 8.572 |
| summarize_scenes | 6.198 |
| synthesize_synopsis | 9.985 |
| make_embedding | 2.537 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 24.268 |
| branch_yolo_total | 8.692 |
| branch_audio_total | 42.893 |
