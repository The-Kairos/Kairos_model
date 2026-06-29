# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 19:29:02 UTC | Uv9Gqkugn0A_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 73.968 | 0.769 | 27.465 | 6.691 | 4.413 | 11.476 | 1.569 |

## 2026-06-25 19:29:02 UTC | Uv9Gqkugn0A_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/Uv9Gqkugn0A_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `73.968` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.769 |
| save_clips | - |
| sample_frames | 0.244 |
| caption_frames | 12.008 |
| sample_fps | 1.804 |
| detect_object_yolo | 6.143 |
| audio_scan | 10.725 |
| asr_timings | 9.354 |
| ast_timings | 7.378 |
| describe_scenes | 6.691 |
| summarize_scenes | 4.413 |
| synthesize_synopsis | 11.476 |
| make_embedding | 1.569 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 12.258 |
| branch_yolo_total | 7.952 |
| branch_audio_total | 27.465 |
