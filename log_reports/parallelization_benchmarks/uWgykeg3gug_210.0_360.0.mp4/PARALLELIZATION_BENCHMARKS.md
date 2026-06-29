# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 01:03:31 UTC | uWgykeg3gug_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 97.294 | 0.762 | 40.617 | 6.341 | 5.386 | 8.267 | 2.244 |

## 2026-06-27 01:03:31 UTC | uWgykeg3gug_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/uWgykeg3gug_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `97.294` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.762 |
| save_clips | - |
| sample_frames | 0.645 |
| caption_frames | 22.658 |
| sample_fps | 2.051 |
| detect_object_yolo | 6.894 |
| audio_scan | 16.009 |
| asr_timings | 8.871 |
| ast_timings | 15.729 |
| describe_scenes | 6.341 |
| summarize_scenes | 5.386 |
| synthesize_synopsis | 8.267 |
| make_embedding | 2.244 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 23.309 |
| branch_yolo_total | 8.952 |
| branch_audio_total | 40.617 |
