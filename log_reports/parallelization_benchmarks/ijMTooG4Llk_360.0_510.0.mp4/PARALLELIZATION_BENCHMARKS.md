# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 09:26:22 UTC | ijMTooG4Llk_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 171.272 | 0.808 | 55.705 | 17.301 | 13.240 | 17.536 | 4.147 |

## 2026-06-26 09:26:22 UTC | ijMTooG4Llk_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/ijMTooG4Llk_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `171.272` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.808 |
| save_clips | - |
| sample_frames | 1.472 |
| caption_frames | 47.529 |
| sample_fps | 2.454 |
| detect_object_yolo | 9.667 |
| audio_scan | 9.743 |
| asr_timings | 10.152 |
| ast_timings | 35.802 |
| describe_scenes | 17.301 |
| summarize_scenes | 13.240 |
| synthesize_synopsis | 17.536 |
| make_embedding | 4.147 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 49.008 |
| branch_yolo_total | 12.127 |
| branch_audio_total | 55.705 |
