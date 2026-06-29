# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 20:46:46 UTC | WbYohTnOUd8_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 185.944 | 0.728 | 71.998 | 21.334 | 10.010 | 9.870 | 4.734 |

## 2026-06-25 20:46:46 UTC | WbYohTnOUd8_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/WbYohTnOUd8_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `185.944` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.728 |
| save_clips | - |
| sample_frames | 1.624 |
| caption_frames | 51.192 |
| sample_fps | 2.366 |
| detect_object_yolo | 10.686 |
| audio_scan | 11.778 |
| asr_timings | 21.063 |
| ast_timings | 39.149 |
| describe_scenes | 21.334 |
| summarize_scenes | 10.010 |
| synthesize_synopsis | 9.870 |
| make_embedding | 4.734 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 52.822 |
| branch_yolo_total | 13.058 |
| branch_audio_total | 71.998 |
