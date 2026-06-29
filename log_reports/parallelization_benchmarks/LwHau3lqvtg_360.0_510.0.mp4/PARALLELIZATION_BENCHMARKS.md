# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 07:54:18 UTC | LwHau3lqvtg_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 153.411 | 0.773 | 69.572 | 14.554 | 18.871 | 11.385 | 2.326 |

## 2026-06-25 07:54:18 UTC | LwHau3lqvtg_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/LwHau3lqvtg_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `153.411` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.773 |
| save_clips | - |
| sample_frames | 0.549 |
| caption_frames | 24.625 |
| sample_fps | 1.962 |
| detect_object_yolo | 7.385 |
| audio_scan | 14.854 |
| asr_timings | 38.991 |
| ast_timings | 15.718 |
| describe_scenes | 14.554 |
| summarize_scenes | 18.871 |
| synthesize_synopsis | 11.385 |
| make_embedding | 2.326 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 25.180 |
| branch_yolo_total | 9.353 |
| branch_audio_total | 69.572 |
