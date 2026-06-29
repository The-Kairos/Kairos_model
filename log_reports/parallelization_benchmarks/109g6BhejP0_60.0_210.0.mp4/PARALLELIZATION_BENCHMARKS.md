# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-21 09:05:55 UTC | 109g6BhejP0_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 116.727 | 1.590 | 49.675 | 8.091 | 6.162 | 8.392 | 3.212 |
| 2026-06-21 20:54:06 UTC | 109g6BhejP0_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 0.060 | - | - | - | - | - | - |
| 2026-06-22 13:13:29 UTC | 109g6BhejP0_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 165.539 | 1.552 | 49.985 | 16.392 | 17.911 | 23.240 | 3.073 |

## 2026-06-21 09:05:55 UTC | 109g6BhejP0_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/109g6BhejP0_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `116.727` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.590 |
| save_clips | - |
| sample_frames | 3.005 |
| caption_frames | 22.033 |
| sample_fps | 5.844 |
| detect_object_yolo | 7.507 |
| audio_scan | 16.096 |
| asr_timings | 9.990 |
| ast_timings | 23.580 |
| describe_scenes | 8.091 |
| summarize_scenes | 6.162 |
| synthesize_synopsis | 8.392 |
| make_embedding | 3.212 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 25.043 |
| branch_yolo_total | 13.357 |
| branch_audio_total | 49.675 |

## 2026-06-21 20:54:06 UTC | 109g6BhejP0_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/109g6BhejP0_60.0_210.0.mp4`
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

## 2026-06-22 13:13:29 UTC | 109g6BhejP0_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/109g6BhejP0_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `165.539` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.552 |
| save_clips | - |
| sample_frames | 2.970 |
| caption_frames | 34.808 |
| sample_fps | 5.928 |
| detect_object_yolo | 8.241 |
| audio_scan | 15.130 |
| asr_timings | 10.452 |
| ast_timings | 24.395 |
| describe_scenes | 16.392 |
| summarize_scenes | 17.911 |
| synthesize_synopsis | 23.240 |
| make_embedding | 3.073 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 37.784 |
| branch_yolo_total | 14.175 |
| branch_audio_total | 49.985 |
