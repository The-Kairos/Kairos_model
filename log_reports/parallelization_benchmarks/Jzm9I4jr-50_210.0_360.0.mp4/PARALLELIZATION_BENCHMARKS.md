# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 05:33:39 UTC | Jzm9I4jr-50_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 137.017 | 0.664 | 43.771 | 11.308 | 8.274 | 16.040 | 3.942 |

## 2026-06-25 05:33:39 UTC | Jzm9I4jr-50_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/Jzm9I4jr-50_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `137.017` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.664 |
| save_clips | - |
| sample_frames | 0.970 |
| caption_frames | 39.282 |
| sample_fps | 2.114 |
| detect_object_yolo | 9.215 |
| audio_scan | 6.477 |
| asr_timings | 10.208 |
| ast_timings | 27.077 |
| describe_scenes | 11.308 |
| summarize_scenes | 8.274 |
| synthesize_synopsis | 16.040 |
| make_embedding | 3.942 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 40.258 |
| branch_yolo_total | 11.335 |
| branch_audio_total | 43.771 |
