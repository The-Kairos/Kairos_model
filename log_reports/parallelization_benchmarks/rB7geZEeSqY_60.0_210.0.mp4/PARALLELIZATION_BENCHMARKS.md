# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 18:33:36 UTC | rB7geZEeSqY_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 133.015 | 0.807 | 50.822 | 13.960 | 7.733 | 10.932 | 3.040 |

## 2026-06-26 18:33:36 UTC | rB7geZEeSqY_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/rB7geZEeSqY_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `133.015` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.807 |
| save_clips | - |
| sample_frames | 0.945 |
| caption_frames | 33.073 |
| sample_fps | 2.172 |
| detect_object_yolo | 8.036 |
| audio_scan | 15.207 |
| asr_timings | 11.936 |
| ast_timings | 23.670 |
| describe_scenes | 13.960 |
| summarize_scenes | 7.733 |
| synthesize_synopsis | 10.932 |
| make_embedding | 3.040 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 34.023 |
| branch_yolo_total | 10.215 |
| branch_audio_total | 50.822 |
