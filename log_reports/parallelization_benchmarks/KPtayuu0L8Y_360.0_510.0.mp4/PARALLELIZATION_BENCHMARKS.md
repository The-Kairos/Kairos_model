# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 06:33:16 UTC | KPtayuu0L8Y_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 159.906 | 0.625 | 41.869 | 16.107 | 33.389 | 24.694 | 2.801 |

## 2026-06-25 06:33:16 UTC | KPtayuu0L8Y_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/KPtayuu0L8Y_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `159.906` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.625 |
| save_clips | - |
| sample_frames | 0.732 |
| caption_frames | 29.104 |
| sample_fps | 1.836 |
| detect_object_yolo | 7.361 |
| audio_scan | 12.417 |
| asr_timings | 8.519 |
| ast_timings | 20.924 |
| describe_scenes | 16.107 |
| summarize_scenes | 33.389 |
| synthesize_synopsis | 24.694 |
| make_embedding | 2.801 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 29.842 |
| branch_yolo_total | 9.203 |
| branch_audio_total | 41.869 |
