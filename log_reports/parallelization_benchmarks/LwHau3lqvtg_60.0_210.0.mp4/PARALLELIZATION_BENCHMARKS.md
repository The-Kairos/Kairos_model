# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 07:57:05 UTC | LwHau3lqvtg_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 165.492 | 0.805 | 89.273 | 6.392 | 10.764 | 18.939 | 2.539 |

## 2026-06-25 07:57:05 UTC | LwHau3lqvtg_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/LwHau3lqvtg_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `165.492` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.805 |
| save_clips | - |
| sample_frames | 0.681 |
| caption_frames | 25.440 |
| sample_fps | 2.063 |
| detect_object_yolo | 7.144 |
| audio_scan | 13.946 |
| asr_timings | 57.068 |
| ast_timings | 18.250 |
| describe_scenes | 6.392 |
| summarize_scenes | 10.764 |
| synthesize_synopsis | 18.939 |
| make_embedding | 2.539 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 26.126 |
| branch_yolo_total | 9.213 |
| branch_audio_total | 89.273 |
