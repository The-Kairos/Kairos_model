# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 17:57:00 UTC | 9V7Jp9K_3AE_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 189.793 | 0.700 | 84.452 | 17.052 | 9.189 | 15.904 | 3.847 |

## 2026-06-24 17:57:00 UTC | 9V7Jp9K_3AE_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/9V7Jp9K_3AE_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `189.793` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.700 |
| save_clips | - |
| sample_frames | 1.111 |
| caption_frames | 44.481 |
| sample_fps | 2.184 |
| detect_object_yolo | 9.402 |
| audio_scan | 14.022 |
| asr_timings | 37.219 |
| ast_timings | 33.202 |
| describe_scenes | 17.052 |
| summarize_scenes | 9.189 |
| synthesize_synopsis | 15.904 |
| make_embedding | 3.847 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 45.598 |
| branch_yolo_total | 11.592 |
| branch_audio_total | 84.452 |
