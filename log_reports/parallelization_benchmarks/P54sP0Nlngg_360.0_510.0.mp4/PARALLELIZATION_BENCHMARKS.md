# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 12:49:42 UTC | P54sP0Nlngg_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 195.430 | 0.790 | 54.288 | 19.527 | 28.022 | 31.157 | 3.715 |

## 2026-06-25 12:49:42 UTC | P54sP0Nlngg_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/P54sP0Nlngg_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `195.430` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.790 |
| save_clips | - |
| sample_frames | 1.096 |
| caption_frames | 43.506 |
| sample_fps | 2.305 |
| detect_object_yolo | 9.563 |
| audio_scan | 11.213 |
| asr_timings | 12.446 |
| ast_timings | 30.620 |
| describe_scenes | 19.527 |
| summarize_scenes | 28.022 |
| synthesize_synopsis | 31.157 |
| make_embedding | 3.715 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 44.608 |
| branch_yolo_total | 11.874 |
| branch_audio_total | 54.288 |
