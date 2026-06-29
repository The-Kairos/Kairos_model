# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-21 20:54:05 UTC | 109g6BhejP0_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 0.060 | - | - | - | - | - | - |
| 2026-06-22 13:10:42 UTC | 109g6BhejP0_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 123.985 | 0.515 | 40.940 | 10.918 | 12.885 | 20.656 | 2.527 |

## 2026-06-21 20:54:05 UTC | 109g6BhejP0_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/109g6BhejP0_360.0_510.0.mp4`
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

## 2026-06-22 13:10:42 UTC | 109g6BhejP0_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/109g6BhejP0_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `123.985` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.515 |
| save_clips | - |
| sample_frames | 0.545 |
| caption_frames | 26.052 |
| sample_fps | 1.557 |
| detect_object_yolo | 6.008 |
| audio_scan | 14.234 |
| asr_timings | 8.394 |
| ast_timings | 18.303 |
| describe_scenes | 10.918 |
| summarize_scenes | 12.885 |
| synthesize_synopsis | 20.656 |
| make_embedding | 2.527 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 26.603 |
| branch_yolo_total | 7.570 |
| branch_audio_total | 40.940 |
