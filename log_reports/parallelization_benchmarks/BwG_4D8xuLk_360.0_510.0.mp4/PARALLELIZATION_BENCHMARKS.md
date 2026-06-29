# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 19:57:04 UTC | BwG_4D8xuLk_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 157.085 | 0.817 | 49.575 | 12.156 | 12.776 | 28.566 | 3.343 |

## 2026-06-24 19:57:04 UTC | BwG_4D8xuLk_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/BwG_4D8xuLk_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `157.085` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.817 |
| save_clips | - |
| sample_frames | 1.010 |
| caption_frames | 36.558 |
| sample_fps | 2.211 |
| detect_object_yolo | 8.646 |
| audio_scan | 12.833 |
| asr_timings | 9.779 |
| ast_timings | 26.955 |
| describe_scenes | 12.156 |
| summarize_scenes | 12.776 |
| synthesize_synopsis | 28.566 |
| make_embedding | 3.343 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 37.574 |
| branch_yolo_total | 10.863 |
| branch_audio_total | 49.575 |
