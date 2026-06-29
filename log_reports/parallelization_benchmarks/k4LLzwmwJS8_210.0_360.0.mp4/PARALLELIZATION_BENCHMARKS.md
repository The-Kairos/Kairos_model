# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 12:44:14 UTC | k4LLzwmwJS8_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 223.041 | 0.841 | 62.847 | 29.193 | 25.884 | 28.428 | 5.096 |

## 2026-06-26 12:44:14 UTC | k4LLzwmwJS8_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/k4LLzwmwJS8_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `223.041` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.841 |
| save_clips | - |
| sample_frames | 1.375 |
| caption_frames | 54.999 |
| sample_fps | 2.435 |
| detect_object_yolo | 10.530 |
| audio_scan | 11.882 |
| asr_timings | 9.299 |
| ast_timings | 41.659 |
| describe_scenes | 29.193 |
| summarize_scenes | 25.884 |
| synthesize_synopsis | 28.428 |
| make_embedding | 5.096 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 56.380 |
| branch_yolo_total | 12.970 |
| branch_audio_total | 62.847 |
