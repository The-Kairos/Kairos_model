# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 23:05:23 UTC | DtLH2de0Wwc_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 178.734 | 0.802 | 89.562 | 9.966 | 12.660 | 15.782 | 3.470 |

## 2026-06-24 23:05:23 UTC | DtLH2de0Wwc_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/DtLH2de0Wwc_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `178.734` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.802 |
| save_clips | - |
| sample_frames | 1.001 |
| caption_frames | 33.868 |
| sample_fps | 2.206 |
| detect_object_yolo | 8.015 |
| audio_scan | 13.998 |
| asr_timings | 48.636 |
| ast_timings | 26.920 |
| describe_scenes | 9.966 |
| summarize_scenes | 12.660 |
| synthesize_synopsis | 15.782 |
| make_embedding | 3.470 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 34.876 |
| branch_yolo_total | 10.227 |
| branch_audio_total | 89.562 |
