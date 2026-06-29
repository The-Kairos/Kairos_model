# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 22:56:33 UTC | Zoae1zkIgIg_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 149.195 | 0.658 | 54.027 | 11.685 | 19.089 | 6.425 | 3.580 |

## 2026-06-25 22:56:33 UTC | Zoae1zkIgIg_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/Zoae1zkIgIg_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `149.195` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.658 |
| save_clips | - |
| sample_frames | 0.840 |
| caption_frames | 40.616 |
| sample_fps | 2.014 |
| detect_object_yolo | 8.853 |
| audio_scan | 13.857 |
| asr_timings | 9.664 |
| ast_timings | 30.498 |
| describe_scenes | 11.685 |
| summarize_scenes | 19.089 |
| synthesize_synopsis | 6.425 |
| make_embedding | 3.580 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 41.463 |
| branch_yolo_total | 10.873 |
| branch_audio_total | 54.027 |
