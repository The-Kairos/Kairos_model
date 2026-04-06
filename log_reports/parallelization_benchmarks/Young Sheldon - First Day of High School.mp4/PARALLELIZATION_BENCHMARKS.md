# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-04-06 07:05:09 UTC | Young Sheldon - First Day of High School.mp4 | semi_parallel | gemini | gemini-embedding-001 | 161.565 | 1.514 | 41.046 | 34.601 | 10.179 | 9.483 | 2.515 |
| 2026-04-06 07:07:31 UTC | Young Sheldon - First Day of High School.mp4 | parallel | gemini | gemini-embedding-001 | 128.822 | 1.469 | 71.880 | 34.820 | 9.432 | 8.521 | 2.535 |

## 2026-04-06 07:05:09 UTC | Young Sheldon - First Day of High School.mp4 | semi_parallel

- Video path: `Videos/Young Sheldon - First Day of High School.mp4`
- Low memory mode: `False`
- Debug: `False`
- Quiet: `False`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `161.565` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.514 |
| save_clips | - |
| sample_frames | 3.369 |
| caption_frames | 37.677 |
| sample_fps | 18.396 |
| detect_object_yolo | 11.559 |
| audio_scan | 14.268 |
| asr_timings | 10.030 |
| ast_timings | 7.798 |
| describe_scenes | 34.601 |
| summarize_scenes | 10.179 |
| synthesize_synopsis | 9.483 |
| make_embedding | 2.515 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 41.046 |
| branch_yolo_total | 29.955 |
| branch_audio_total | 32.096 |

## 2026-04-06 07:07:31 UTC | Young Sheldon - First Day of High School.mp4 | parallel

- Video path: `Videos/Young Sheldon - First Day of High School.mp4`
- Low memory mode: `False`
- Debug: `False`
- Quiet: `False`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `128.822` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.469 |
| save_clips | - |
| sample_frames | 4.680 |
| caption_frames | 67.200 |
| sample_fps | 28.293 |
| detect_object_yolo | 14.580 |
| audio_scan | 16.128 |
| asr_timings | 15.896 |
| ast_timings | 8.658 |
| describe_scenes | 34.820 |
| summarize_scenes | 9.432 |
| synthesize_synopsis | 8.521 |
| make_embedding | 2.535 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 71.880 |
| branch_yolo_total | 42.873 |
| branch_audio_total | 32.024 |
