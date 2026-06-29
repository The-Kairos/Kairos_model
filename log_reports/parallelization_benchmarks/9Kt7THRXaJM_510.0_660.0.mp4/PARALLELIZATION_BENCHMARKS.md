# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 17:48:29 UTC | 9Kt7THRXaJM_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 176.251 | 0.805 | 68.533 | 17.999 | 10.702 | 14.827 | 3.790 |

## 2026-06-24 17:48:29 UTC | 9Kt7THRXaJM_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/9Kt7THRXaJM_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `176.251` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.805 |
| save_clips | - |
| sample_frames | 1.240 |
| caption_frames | 45.647 |
| sample_fps | 2.336 |
| detect_object_yolo | 8.987 |
| audio_scan | 13.846 |
| asr_timings | 21.971 |
| ast_timings | 32.708 |
| describe_scenes | 17.999 |
| summarize_scenes | 10.702 |
| synthesize_synopsis | 14.827 |
| make_embedding | 3.790 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 46.893 |
| branch_yolo_total | 11.329 |
| branch_audio_total | 68.533 |
