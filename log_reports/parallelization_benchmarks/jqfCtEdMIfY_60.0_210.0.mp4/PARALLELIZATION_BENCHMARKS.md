# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 12:06:44 UTC | jqfCtEdMIfY_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 247.443 | 0.792 | 100.172 | 20.979 | 37.918 | 18.472 | 4.442 |

## 2026-06-26 12:06:44 UTC | jqfCtEdMIfY_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/jqfCtEdMIfY_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `247.443` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.792 |
| save_clips | - |
| sample_frames | 1.463 |
| caption_frames | 49.113 |
| sample_fps | 2.416 |
| detect_object_yolo | 10.258 |
| audio_scan | 15.025 |
| asr_timings | 47.320 |
| ast_timings | 37.819 |
| describe_scenes | 20.979 |
| summarize_scenes | 37.918 |
| synthesize_synopsis | 18.472 |
| make_embedding | 4.442 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 50.582 |
| branch_yolo_total | 12.681 |
| branch_audio_total | 100.172 |
