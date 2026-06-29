# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-21 20:54:04 UTC | 109g6BhejP0_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 0.060 | - | - | - | - | - | - |
| 2026-06-22 13:08:37 UTC | 109g6BhejP0_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 170.849 | 0.627 | 51.928 | 19.190 | 10.352 | 33.695 | 3.562 |

## 2026-06-21 20:54:04 UTC | 109g6BhejP0_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/109g6BhejP0_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `0.060` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | - |
| save_clips | - |
| sample_frames | - |
| caption_frames | - |
| sample_fps | - |
| detect_object_yolo | - |
| audio_scan | - |
| asr_timings | - |
| ast_timings | - |
| describe_scenes | - |
| summarize_scenes | - |
| synthesize_synopsis | - |
| make_embedding | - |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | - |
| branch_yolo_total | - |
| branch_audio_total | - |

## 2026-06-22 13:08:37 UTC | 109g6BhejP0_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/109g6BhejP0_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `170.849` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.627 |
| save_clips | - |
| sample_frames | 0.929 |
| caption_frames | 38.695 |
| sample_fps | 2.010 |
| detect_object_yolo | 8.476 |
| audio_scan | 11.663 |
| asr_timings | 10.199 |
| ast_timings | 30.057 |
| describe_scenes | 19.190 |
| summarize_scenes | 10.352 |
| synthesize_synopsis | 33.695 |
| make_embedding | 3.562 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 39.630 |
| branch_yolo_total | 10.492 |
| branch_audio_total | 51.928 |
