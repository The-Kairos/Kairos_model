# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 23:47:37 UTC | Ejf572xg02M_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 84.078 | 0.670 | 39.911 | 6.048 | 5.222 | 10.140 | 1.582 |

## 2026-06-24 23:47:37 UTC | Ejf572xg02M_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/Ejf572xg02M_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `84.078` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.670 |
| save_clips | - |
| sample_frames | 0.228 |
| caption_frames | 11.732 |
| sample_fps | 1.573 |
| detect_object_yolo | 5.591 |
| audio_scan | 10.174 |
| asr_timings | 22.211 |
| ast_timings | 7.518 |
| describe_scenes | 6.048 |
| summarize_scenes | 5.222 |
| synthesize_synopsis | 10.140 |
| make_embedding | 1.582 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 11.966 |
| branch_yolo_total | 7.169 |
| branch_audio_total | 39.911 |
