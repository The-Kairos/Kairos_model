# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 00:34:43 UTC | uCZAfLBvPVo_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 177.787 | 0.654 | 71.363 | 13.868 | 9.988 | 9.553 | 4.423 |

## 2026-06-27 00:34:43 UTC | uCZAfLBvPVo_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/uCZAfLBvPVo_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `177.787` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.654 |
| save_clips | - |
| sample_frames | 1.697 |
| caption_frames | 52.000 |
| sample_fps | 2.371 |
| detect_object_yolo | 10.417 |
| audio_scan | 13.940 |
| asr_timings | 18.377 |
| ast_timings | 39.038 |
| describe_scenes | 13.868 |
| summarize_scenes | 9.988 |
| synthesize_synopsis | 9.553 |
| make_embedding | 4.423 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 53.703 |
| branch_yolo_total | 12.794 |
| branch_audio_total | 71.363 |
