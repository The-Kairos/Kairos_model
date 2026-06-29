# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-23 15:56:52 UTC | 3G2WCQnH6Nk_660.0_810.0.mp4 | sequential | gemini | gemini-embedding-001 | 181.985 | 0.767 | 63.256 | 24.905 | 12.602 | 29.694 | 3.351 |
| 2026-06-24 09:53:07 UTC | 3G2WCQnH6Nk_660.0_810.0.mp4 | sequential | gemini | gemini-embedding-001 | 187.474 | 0.786 | 70.262 | 26.514 | 15.700 | 22.216 | 3.433 |

## 2026-06-23 15:56:52 UTC | 3G2WCQnH6Nk_660.0_810.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/3G2WCQnH6Nk_660.0_810.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `181.985` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.767 |
| save_clips | - |
| sample_frames | 1.073 |
| caption_frames | 34.417 |
| sample_fps | 2.193 |
| detect_object_yolo | 8.353 |
| audio_scan | 13.747 |
| asr_timings | 24.032 |
| ast_timings | 25.468 |
| describe_scenes | 24.905 |
| summarize_scenes | 12.602 |
| synthesize_synopsis | 29.694 |
| make_embedding | 3.351 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 35.496 |
| branch_yolo_total | 10.551 |
| branch_audio_total | 63.256 |

## 2026-06-24 09:53:07 UTC | 3G2WCQnH6Nk_660.0_810.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/3G2WCQnH6Nk_660.0_810.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `187.474` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.786 |
| save_clips | - |
| sample_frames | 1.092 |
| caption_frames | 35.431 |
| sample_fps | 2.260 |
| detect_object_yolo | 8.372 |
| audio_scan | 13.874 |
| asr_timings | 30.675 |
| ast_timings | 25.705 |
| describe_scenes | 26.514 |
| summarize_scenes | 15.700 |
| synthesize_synopsis | 22.216 |
| make_embedding | 3.433 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 36.529 |
| branch_yolo_total | 10.638 |
| branch_audio_total | 70.262 |
