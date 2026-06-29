# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 05:05:42 UTC | yQ5wwBumNG8_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 147.515 | 0.645 | 67.135 | 9.668 | 7.167 | 7.041 | 3.529 |

## 2026-06-27 05:05:42 UTC | yQ5wwBumNG8_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/yQ5wwBumNG8_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `147.515` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.645 |
| save_clips | - |
| sample_frames | 0.990 |
| caption_frames | 39.054 |
| sample_fps | 2.133 |
| detect_object_yolo | 8.699 |
| audio_scan | 7.720 |
| asr_timings | 29.531 |
| ast_timings | 29.874 |
| describe_scenes | 9.668 |
| summarize_scenes | 7.167 |
| synthesize_synopsis | 7.041 |
| make_embedding | 3.529 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 40.050 |
| branch_yolo_total | 10.838 |
| branch_audio_total | 67.135 |
