# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 02:20:26 UTC | GxqR6p8r6z0_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 203.514 | 0.702 | 71.329 | 15.486 | 17.172 | 9.332 | 6.572 |

## 2026-06-25 02:20:26 UTC | GxqR6p8r6z0_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/GxqR6p8r6z0_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `203.514` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.702 |
| save_clips | - |
| sample_frames | 1.728 |
| caption_frames | 64.827 |
| sample_fps | 2.512 |
| detect_object_yolo | 12.398 |
| audio_scan | 10.715 |
| asr_timings | 10.984 |
| ast_timings | 49.622 |
| describe_scenes | 15.486 |
| summarize_scenes | 17.172 |
| synthesize_synopsis | 9.332 |
| make_embedding | 6.572 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 66.561 |
| branch_yolo_total | 14.917 |
| branch_audio_total | 71.329 |
