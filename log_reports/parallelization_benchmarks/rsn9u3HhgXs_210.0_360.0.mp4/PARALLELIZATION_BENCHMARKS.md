# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 19:04:58 UTC | rsn9u3HhgXs_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 212.218 | 0.803 | 83.096 | 23.452 | 26.054 | 17.822 | 3.888 |

## 2026-06-26 19:04:58 UTC | rsn9u3HhgXs_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/rsn9u3HhgXs_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `212.218` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.803 |
| save_clips | - |
| sample_frames | 1.316 |
| caption_frames | 42.293 |
| sample_fps | 2.317 |
| detect_object_yolo | 9.663 |
| audio_scan | 14.008 |
| asr_timings | 36.306 |
| ast_timings | 32.773 |
| describe_scenes | 23.452 |
| summarize_scenes | 26.054 |
| synthesize_synopsis | 17.822 |
| make_embedding | 3.888 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 43.614 |
| branch_yolo_total | 11.987 |
| branch_audio_total | 83.096 |
