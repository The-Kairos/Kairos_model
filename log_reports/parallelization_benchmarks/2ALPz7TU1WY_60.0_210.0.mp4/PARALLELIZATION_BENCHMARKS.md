# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-23 14:23:07 UTC | 2ALPz7TU1WY_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 186.502 | 0.768 | 53.179 | 22.348 | 24.375 | 25.049 | 3.907 |
| 2026-06-27 15:40:44 UTC | 2ALPz7TU1WY_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 148.655 | 0.780 | 53.705 | 9.455 | 11.777 | 11.886 | 3.908 |

## 2026-06-23 14:23:07 UTC | 2ALPz7TU1WY_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/2ALPz7TU1WY_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `186.502` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.768 |
| save_clips | - |
| sample_frames | 0.975 |
| caption_frames | 43.377 |
| sample_fps | 2.172 |
| detect_object_yolo | 8.987 |
| audio_scan | 10.574 |
| asr_timings | 9.959 |
| ast_timings | 32.637 |
| describe_scenes | 22.348 |
| summarize_scenes | 24.375 |
| synthesize_synopsis | 25.049 |
| make_embedding | 3.907 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 44.357 |
| branch_yolo_total | 11.165 |
| branch_audio_total | 53.179 |

## 2026-06-27 15:40:44 UTC | 2ALPz7TU1WY_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/2ALPz7TU1WY_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `148.655` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.780 |
| save_clips | - |
| sample_frames | 0.979 |
| caption_frames | 43.373 |
| sample_fps | 2.233 |
| detect_object_yolo | 9.158 |
| audio_scan | 10.621 |
| asr_timings | 10.048 |
| ast_timings | 33.029 |
| describe_scenes | 9.455 |
| summarize_scenes | 11.777 |
| synthesize_synopsis | 11.886 |
| make_embedding | 3.908 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 44.357 |
| branch_yolo_total | 11.397 |
| branch_audio_total | 53.705 |
