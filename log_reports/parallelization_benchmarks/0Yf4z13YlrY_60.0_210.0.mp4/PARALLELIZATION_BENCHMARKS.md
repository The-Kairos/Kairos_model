# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-21 20:53:48 UTC | 0Yf4z13YlrY_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 0.059 | - | - | - | - | - | - |
| 2026-06-22 12:30:35 UTC | 0Yf4z13YlrY_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 163.183 | 0.775 | 48.504 | 21.877 | 9.741 | 32.821 | 3.084 |

## 2026-06-21 20:53:48 UTC | 0Yf4z13YlrY_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/0Yf4z13YlrY_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `0.059` sec

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

## 2026-06-22 12:30:35 UTC | 0Yf4z13YlrY_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/0Yf4z13YlrY_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `163.183` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.775 |
| save_clips | - |
| sample_frames | 0.974 |
| caption_frames | 33.940 |
| sample_fps | 2.162 |
| detect_object_yolo | 7.939 |
| audio_scan | 15.928 |
| asr_timings | 8.902 |
| ast_timings | 23.665 |
| describe_scenes | 21.877 |
| summarize_scenes | 9.741 |
| synthesize_synopsis | 32.821 |
| make_embedding | 3.084 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 34.919 |
| branch_yolo_total | 10.107 |
| branch_audio_total | 48.504 |
