# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 05:40:29 UTC | Jzm9I4jr-50_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 213.217 | 0.677 | 61.456 | 21.169 | 16.066 | 31.792 | 5.340 |

## 2026-06-25 05:40:29 UTC | Jzm9I4jr-50_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/Jzm9I4jr-50_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `213.217` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.677 |
| save_clips | - |
| sample_frames | 1.483 |
| caption_frames | 59.865 |
| sample_fps | 2.432 |
| detect_object_yolo | 11.514 |
| audio_scan | 6.475 |
| asr_timings | 11.270 |
| ast_timings | 43.702 |
| describe_scenes | 21.169 |
| summarize_scenes | 16.066 |
| synthesize_synopsis | 31.792 |
| make_embedding | 5.340 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 61.355 |
| branch_yolo_total | 13.952 |
| branch_audio_total | 61.456 |
