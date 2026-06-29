# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 19:58:40 UTC | sPm32nQ_lc0_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 159.525 | 0.801 | 53.154 | 15.357 | 20.525 | 10.830 | 3.516 |

## 2026-06-26 19:58:40 UTC | sPm32nQ_lc0_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/sPm32nQ_lc0_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `159.525` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.801 |
| save_clips | - |
| sample_frames | 1.233 |
| caption_frames | 41.512 |
| sample_fps | 2.292 |
| detect_object_yolo | 8.859 |
| audio_scan | 14.197 |
| asr_timings | 9.651 |
| ast_timings | 29.296 |
| describe_scenes | 15.357 |
| summarize_scenes | 20.525 |
| synthesize_synopsis | 10.830 |
| make_embedding | 3.516 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 42.751 |
| branch_yolo_total | 11.157 |
| branch_audio_total | 53.154 |
