# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 22:44:35 UTC | ZhIAoqsl0GY_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 157.834 | 0.665 | 53.948 | 12.730 | 8.166 | 18.585 | 3.843 |

## 2026-06-25 22:44:35 UTC | ZhIAoqsl0GY_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/ZhIAoqsl0GY_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `157.834` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.665 |
| save_clips | - |
| sample_frames | 1.503 |
| caption_frames | 45.154 |
| sample_fps | 2.306 |
| detect_object_yolo | 9.509 |
| audio_scan | 8.480 |
| asr_timings | 11.606 |
| ast_timings | 33.853 |
| describe_scenes | 12.730 |
| summarize_scenes | 8.166 |
| synthesize_synopsis | 18.585 |
| make_embedding | 3.843 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 46.662 |
| branch_yolo_total | 11.821 |
| branch_audio_total | 53.948 |
