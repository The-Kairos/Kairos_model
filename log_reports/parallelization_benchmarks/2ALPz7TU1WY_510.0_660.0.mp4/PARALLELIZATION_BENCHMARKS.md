# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-23 14:20:00 UTC | 2ALPz7TU1WY_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 167.185 | 0.779 | 45.979 | 27.527 | 16.117 | 27.357 | 3.102 |
| 2026-06-27 15:38:14 UTC | 2ALPz7TU1WY_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 118.379 | 0.768 | 46.128 | 8.440 | 4.899 | 9.498 | 3.108 |

## 2026-06-23 14:20:00 UTC | 2ALPz7TU1WY_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/2ALPz7TU1WY_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `167.185` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.779 |
| save_clips | - |
| sample_frames | 0.811 |
| caption_frames | 34.192 |
| sample_fps | 2.102 |
| detect_object_yolo | 7.846 |
| audio_scan | 11.657 |
| asr_timings | 10.616 |
| ast_timings | 23.697 |
| describe_scenes | 27.527 |
| summarize_scenes | 16.117 |
| synthesize_synopsis | 27.357 |
| make_embedding | 3.102 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 35.010 |
| branch_yolo_total | 9.953 |
| branch_audio_total | 45.979 |

## 2026-06-27 15:38:14 UTC | 2ALPz7TU1WY_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/2ALPz7TU1WY_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `118.379` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.768 |
| save_clips | - |
| sample_frames | 0.798 |
| caption_frames | 33.123 |
| sample_fps | 2.123 |
| detect_object_yolo | 8.065 |
| audio_scan | 11.776 |
| asr_timings | 10.372 |
| ast_timings | 23.972 |
| describe_scenes | 8.440 |
| summarize_scenes | 4.899 |
| synthesize_synopsis | 9.498 |
| make_embedding | 3.108 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 33.928 |
| branch_yolo_total | 10.194 |
| branch_audio_total | 46.128 |
