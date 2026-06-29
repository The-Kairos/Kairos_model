# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 22:39:12 UTC | ZhIAoqsl0GY_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 156.956 | 0.671 | 57.853 | 14.538 | 9.802 | 13.222 | 3.859 |

## 2026-06-25 22:39:12 UTC | ZhIAoqsl0GY_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/ZhIAoqsl0GY_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `156.956` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.671 |
| save_clips | - |
| sample_frames | 1.319 |
| caption_frames | 42.933 |
| sample_fps | 2.199 |
| detect_object_yolo | 9.163 |
| audio_scan | 13.743 |
| asr_timings | 10.007 |
| ast_timings | 34.095 |
| describe_scenes | 14.538 |
| summarize_scenes | 9.802 |
| synthesize_synopsis | 13.222 |
| make_embedding | 3.859 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 44.258 |
| branch_yolo_total | 11.367 |
| branch_audio_total | 57.853 |
