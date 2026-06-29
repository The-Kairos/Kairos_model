# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 01:04:57 UTC | uWgykeg3gug_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 84.478 | 0.768 | 35.635 | 4.971 | 4.733 | 8.695 | 2.033 |

## 2026-06-27 01:04:57 UTC | uWgykeg3gug_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/uWgykeg3gug_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `84.478` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.768 |
| save_clips | - |
| sample_frames | 0.434 |
| caption_frames | 17.921 |
| sample_fps | 1.946 |
| detect_object_yolo | 5.926 |
| audio_scan | 12.710 |
| asr_timings | 9.436 |
| ast_timings | 13.481 |
| describe_scenes | 4.971 |
| summarize_scenes | 4.733 |
| synthesize_synopsis | 8.695 |
| make_embedding | 2.033 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 18.361 |
| branch_yolo_total | 7.878 |
| branch_audio_total | 35.635 |
