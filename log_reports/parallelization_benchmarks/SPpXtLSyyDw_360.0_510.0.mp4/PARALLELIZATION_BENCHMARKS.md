# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 17:06:27 UTC | SPpXtLSyyDw_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 128.353 | 0.801 | 50.204 | 10.692 | 9.215 | 11.271 | 3.092 |

## 2026-06-25 17:06:27 UTC | SPpXtLSyyDw_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/SPpXtLSyyDw_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `128.353` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.801 |
| save_clips | - |
| sample_frames | 0.687 |
| caption_frames | 31.356 |
| sample_fps | 2.022 |
| detect_object_yolo | 7.604 |
| audio_scan | 8.546 |
| asr_timings | 17.419 |
| ast_timings | 24.231 |
| describe_scenes | 10.692 |
| summarize_scenes | 9.215 |
| synthesize_synopsis | 11.271 |
| make_embedding | 3.092 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 32.049 |
| branch_yolo_total | 9.632 |
| branch_audio_total | 50.204 |
