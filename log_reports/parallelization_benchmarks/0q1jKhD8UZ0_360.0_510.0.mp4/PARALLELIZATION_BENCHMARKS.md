# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-21 20:53:57 UTC | 0q1jKhD8UZ0_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 0.062 | - | - | - | - | - | - |
| 2026-06-22 12:49:05 UTC | 0q1jKhD8UZ0_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 178.922 | 0.644 | 56.073 | 23.294 | 14.279 | 23.728 | 3.904 |

## 2026-06-21 20:53:57 UTC | 0q1jKhD8UZ0_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/0q1jKhD8UZ0_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `0.062` sec

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

## 2026-06-22 12:49:05 UTC | 0q1jKhD8UZ0_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/0q1jKhD8UZ0_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `178.922` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.644 |
| save_clips | - |
| sample_frames | 0.956 |
| caption_frames | 42.939 |
| sample_fps | 2.118 |
| detect_object_yolo | 9.578 |
| audio_scan | 13.799 |
| asr_timings | 9.886 |
| ast_timings | 32.380 |
| describe_scenes | 23.294 |
| summarize_scenes | 14.279 |
| synthesize_synopsis | 23.728 |
| make_embedding | 3.904 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 43.901 |
| branch_yolo_total | 11.701 |
| branch_audio_total | 56.073 |
