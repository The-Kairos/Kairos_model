# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 04:30:39 UTC | xv_YH57C1MU_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 154.494 | 0.658 | 53.134 | 12.585 | 19.365 | 5.622 | 3.902 |

## 2026-06-27 04:30:39 UTC | xv_YH57C1MU_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/xv_YH57C1MU_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `154.494` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.658 |
| save_clips | - |
| sample_frames | 1.394 |
| caption_frames | 44.551 |
| sample_fps | 2.263 |
| detect_object_yolo | 9.607 |
| audio_scan | 12.969 |
| asr_timings | 8.014 |
| ast_timings | 32.143 |
| describe_scenes | 12.585 |
| summarize_scenes | 19.365 |
| synthesize_synopsis | 5.622 |
| make_embedding | 3.902 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 45.951 |
| branch_yolo_total | 11.876 |
| branch_audio_total | 53.134 |
