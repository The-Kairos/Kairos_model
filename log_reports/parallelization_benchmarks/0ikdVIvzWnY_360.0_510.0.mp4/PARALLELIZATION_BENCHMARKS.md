# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 13:48:03 UTC | 0ikdVIvzWnY_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 118.214 | 0.766 | 48.890 | 8.988 | 5.581 | 8.965 | 2.827 |

## 2026-06-27 13:48:03 UTC | 0ikdVIvzWnY_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/0ikdVIvzWnY_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `118.214` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.766 |
| save_clips | - |
| sample_frames | 0.896 |
| caption_frames | 30.211 |
| sample_fps | 2.165 |
| detect_object_yolo | 7.523 |
| audio_scan | 14.973 |
| asr_timings | 12.434 |
| ast_timings | 21.474 |
| describe_scenes | 8.988 |
| summarize_scenes | 5.581 |
| synthesize_synopsis | 8.965 |
| make_embedding | 2.827 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 31.112 |
| branch_yolo_total | 9.693 |
| branch_audio_total | 48.890 |
