# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 20:58:16 UTC | WgsOaYbE2mw_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 134.855 | 0.806 | 49.400 | 9.484 | 9.386 | 11.723 | 3.521 |

## 2026-06-25 20:58:16 UTC | WgsOaYbE2mw_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/WgsOaYbE2mw_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `134.855` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.806 |
| save_clips | - |
| sample_frames | 1.381 |
| caption_frames | 36.981 |
| sample_fps | 2.298 |
| detect_object_yolo | 8.482 |
| audio_scan | 9.648 |
| asr_timings | 9.380 |
| ast_timings | 30.365 |
| describe_scenes | 9.484 |
| summarize_scenes | 9.386 |
| synthesize_synopsis | 11.723 |
| make_embedding | 3.521 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 38.368 |
| branch_yolo_total | 10.786 |
| branch_audio_total | 49.400 |
