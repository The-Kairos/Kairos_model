# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 16:37:01 UTC | nUWCO8U02U4_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 113.316 | 0.843 | 35.747 | 10.388 | 11.261 | 9.054 | 3.329 |

## 2026-06-27 16:37:01 UTC | nUWCO8U02U4_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/nUWCO8U02U4_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `113.316` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.843 |
| save_clips | - |
| sample_frames | 0.803 |
| caption_frames | 34.937 |
| sample_fps | 2.141 |
| detect_object_yolo | 8.332 |
| audio_scan | 3.801 |
| asr_timings | 0.000 |
| ast_timings | 27.035 |
| describe_scenes | 10.388 |
| summarize_scenes | 11.261 |
| synthesize_synopsis | 9.054 |
| make_embedding | 3.329 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 35.747 |
| branch_yolo_total | 10.479 |
| branch_audio_total | 30.844 |
