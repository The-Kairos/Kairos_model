# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 19:33:39 UTC | Uv9Gqkugn0A_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 178.393 | 0.793 | 62.267 | 21.151 | 9.294 | 10.883 | 5.055 |

## 2026-06-25 19:33:39 UTC | Uv9Gqkugn0A_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/Uv9Gqkugn0A_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `178.393` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.793 |
| save_clips | - |
| sample_frames | 1.828 |
| caption_frames | 52.761 |
| sample_fps | 2.546 |
| detect_object_yolo | 10.417 |
| audio_scan | 14.900 |
| asr_timings | 7.926 |
| ast_timings | 39.432 |
| describe_scenes | 21.151 |
| summarize_scenes | 9.294 |
| synthesize_synopsis | 10.883 |
| make_embedding | 5.055 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 54.595 |
| branch_yolo_total | 12.969 |
| branch_audio_total | 62.267 |
