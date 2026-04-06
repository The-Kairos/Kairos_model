# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-04-06 07:13:01 UTC | Argentina v France Full Penalty Shoot-out.mp4 | semi_parallel | gemini | gemini-embedding-001 | 317.074 | 2.497 | 84.340 | 79.458 | 13.840 | 8.087 | 3.853 |
| 2026-04-06 07:17:21 UTC | Argentina v France Full Penalty Shoot-out.mp4 | parallel | gemini | gemini-embedding-001 | 246.552 | 2.476 | 140.177 | 74.794 | 12.942 | 12.055 | 3.940 |

## 2026-04-06 07:13:01 UTC | Argentina v France Full Penalty Shoot-out.mp4 | semi_parallel

- Video path: `Videos/Argentina v France Full Penalty Shoot-out.mp4`
- Low memory mode: `False`
- Debug: `False`
- Quiet: `False`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `317.074` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 2.497 |
| save_clips | - |
| sample_frames | 5.691 |
| caption_frames | 78.649 |
| sample_fps | 41.228 |
| detect_object_yolo | 25.924 |
| audio_scan | 22.505 |
| asr_timings | 20.562 |
| ast_timings | 14.591 |
| describe_scenes | 79.458 |
| summarize_scenes | 13.840 |
| synthesize_synopsis | 8.087 |
| make_embedding | 3.853 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 84.340 |
| branch_yolo_total | 67.152 |
| branch_audio_total | 57.658 |

## 2026-04-06 07:17:21 UTC | Argentina v France Full Penalty Shoot-out.mp4 | parallel

- Video path: `Videos/Argentina v France Full Penalty Shoot-out.mp4`
- Low memory mode: `False`
- Debug: `False`
- Quiet: `False`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `246.552` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 2.476 |
| save_clips | - |
| sample_frames | 6.771 |
| caption_frames | 133.406 |
| sample_fps | 61.148 |
| detect_object_yolo | 30.717 |
| audio_scan | 26.072 |
| asr_timings | 32.908 |
| ast_timings | 14.956 |
| describe_scenes | 74.794 |
| summarize_scenes | 12.942 |
| synthesize_synopsis | 12.055 |
| make_embedding | 3.940 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 140.177 |
| branch_yolo_total | 91.865 |
| branch_audio_total | 58.980 |
