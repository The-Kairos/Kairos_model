# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 20:17:07 UTC | Vu0Z5BdPKaY_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 199.477 | 0.767 | 70.184 | 17.994 | 20.447 | 10.858 | 5.752 |

## 2026-06-25 20:17:07 UTC | Vu0Z5BdPKaY_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/Vu0Z5BdPKaY_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `199.477` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.767 |
| save_clips | - |
| sample_frames | 1.264 |
| caption_frames | 57.216 |
| sample_fps | 2.404 |
| detect_object_yolo | 11.186 |
| audio_scan | 16.006 |
| asr_timings | 10.464 |
| ast_timings | 43.706 |
| describe_scenes | 17.994 |
| summarize_scenes | 20.447 |
| synthesize_synopsis | 10.858 |
| make_embedding | 5.752 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 58.486 |
| branch_yolo_total | 13.596 |
| branch_audio_total | 70.184 |
