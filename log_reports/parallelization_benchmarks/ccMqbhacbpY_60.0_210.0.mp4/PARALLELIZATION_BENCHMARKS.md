# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 02:35:39 UTC | ccMqbhacbpY_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 150.948 | 0.804 | 56.845 | 11.187 | 9.619 | 11.793 | 3.814 |

## 2026-06-26 02:35:39 UTC | ccMqbhacbpY_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/ccMqbhacbpY_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `150.948` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.804 |
| save_clips | - |
| sample_frames | 1.425 |
| caption_frames | 42.651 |
| sample_fps | 2.365 |
| detect_object_yolo | 9.038 |
| audio_scan | 15.145 |
| asr_timings | 9.309 |
| ast_timings | 32.383 |
| describe_scenes | 11.187 |
| summarize_scenes | 9.619 |
| synthesize_synopsis | 11.793 |
| make_embedding | 3.814 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 44.082 |
| branch_yolo_total | 11.408 |
| branch_audio_total | 56.845 |
