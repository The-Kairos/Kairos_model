# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 03:18:46 UTC | wfIWTD30gCw_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 155.956 | 0.778 | 59.665 | 12.942 | 8.601 | 5.554 | 4.449 |

## 2026-06-27 03:18:46 UTC | wfIWTD30gCw_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/wfIWTD30gCw_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `155.956` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.778 |
| save_clips | - |
| sample_frames | 1.051 |
| caption_frames | 49.102 |
| sample_fps | 2.295 |
| detect_object_yolo | 10.086 |
| audio_scan | 10.837 |
| asr_timings | 10.498 |
| ast_timings | 38.321 |
| describe_scenes | 12.942 |
| summarize_scenes | 8.601 |
| synthesize_synopsis | 5.554 |
| make_embedding | 4.449 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 50.159 |
| branch_yolo_total | 12.387 |
| branch_audio_total | 59.665 |
