# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-21 20:53:47 UTC | 0Yf4z13YlrY_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 0.060 | - | - | - | - | - | - |
| 2026-06-22 12:27:50 UTC | 0Yf4z13YlrY_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 132.100 | 0.683 | 44.251 | 12.679 | 16.007 | 15.610 | 2.825 |

## 2026-06-21 20:53:47 UTC | 0Yf4z13YlrY_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/0Yf4z13YlrY_210.0_360.0.mp4`
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

## 2026-06-22 12:27:50 UTC | 0Yf4z13YlrY_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/0Yf4z13YlrY_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `132.100` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.683 |
| save_clips | - |
| sample_frames | 0.798 |
| caption_frames | 29.456 |
| sample_fps | 1.892 |
| detect_object_yolo | 6.512 |
| audio_scan | 14.481 |
| asr_timings | 8.826 |
| ast_timings | 20.936 |
| describe_scenes | 12.679 |
| summarize_scenes | 16.007 |
| synthesize_synopsis | 15.610 |
| make_embedding | 2.825 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 30.260 |
| branch_yolo_total | 8.410 |
| branch_audio_total | 44.251 |
