# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-23 15:46:05 UTC | 3G2WCQnH6Nk_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 149.070 | 0.785 | 40.500 | 19.581 | 14.804 | 26.265 | 3.103 |
| 2026-06-24 09:42:13 UTC | 3G2WCQnH6Nk_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 143.136 | 0.768 | 40.294 | 12.342 | 23.012 | 21.129 | 3.064 |

## 2026-06-23 15:46:05 UTC | 3G2WCQnH6Nk_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/3G2WCQnH6Nk_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `149.070` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.785 |
| save_clips | - |
| sample_frames | 0.948 |
| caption_frames | 31.684 |
| sample_fps | 2.162 |
| detect_object_yolo | 7.805 |
| audio_scan | 9.642 |
| asr_timings | 6.451 |
| ast_timings | 24.398 |
| describe_scenes | 19.581 |
| summarize_scenes | 14.804 |
| synthesize_synopsis | 26.265 |
| make_embedding | 3.103 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 32.637 |
| branch_yolo_total | 9.973 |
| branch_audio_total | 40.500 |

## 2026-06-24 09:42:13 UTC | 3G2WCQnH6Nk_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/3G2WCQnH6Nk_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `143.136` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.768 |
| save_clips | - |
| sample_frames | 0.941 |
| caption_frames | 30.378 |
| sample_fps | 2.185 |
| detect_object_yolo | 7.647 |
| audio_scan | 9.593 |
| asr_timings | 6.495 |
| ast_timings | 24.197 |
| describe_scenes | 12.342 |
| summarize_scenes | 23.012 |
| synthesize_synopsis | 21.129 |
| make_embedding | 3.064 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 31.324 |
| branch_yolo_total | 9.837 |
| branch_audio_total | 40.294 |
