# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 03:23:33 UTC | wfIWTD30gCw_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 165.815 | 0.766 | 59.939 | 12.894 | 14.436 | 9.700 | 4.401 |

## 2026-06-27 03:23:33 UTC | wfIWTD30gCw_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/wfIWTD30gCw_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `165.815` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.766 |
| save_clips | - |
| sample_frames | 1.189 |
| caption_frames | 48.886 |
| sample_fps | 2.338 |
| detect_object_yolo | 9.873 |
| audio_scan | 10.859 |
| asr_timings | 11.435 |
| ast_timings | 37.637 |
| describe_scenes | 12.894 |
| summarize_scenes | 14.436 |
| synthesize_synopsis | 9.700 |
| make_embedding | 4.401 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 50.081 |
| branch_yolo_total | 12.216 |
| branch_audio_total | 59.939 |
