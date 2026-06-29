# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 08:22:35 UTC | -OuKllHRb04_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 216.998 | 0.828 | 69.269 | 25.283 | 24.848 | 18.821 | 5.588 |

## 2026-06-24 08:22:35 UTC | -OuKllHRb04_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/-OuKllHRb04_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `216.998` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.828 |
| save_clips | - |
| sample_frames | 1.257 |
| caption_frames | 55.994 |
| sample_fps | 2.435 |
| detect_object_yolo | 11.296 |
| audio_scan | 15.042 |
| asr_timings | 11.865 |
| ast_timings | 42.353 |
| describe_scenes | 25.283 |
| summarize_scenes | 24.848 |
| synthesize_synopsis | 18.821 |
| make_embedding | 5.588 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 57.257 |
| branch_yolo_total | 13.736 |
| branch_audio_total | 69.269 |
