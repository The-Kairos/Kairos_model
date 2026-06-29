# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-28 08:14:49 UTC | plsiz20Q_ho_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 106.340 | 0.652 | 39.446 | 7.616 | 9.656 | 10.660 | 2.536 |

## 2026-06-28 08:14:49 UTC | plsiz20Q_ho_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/plsiz20Q_ho_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `106.340` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.652 |
| save_clips | - |
| sample_frames | 0.552 |
| caption_frames | 24.311 |
| sample_fps | 1.855 |
| detect_object_yolo | 7.590 |
| audio_scan | 15.068 |
| asr_timings | 8.685 |
| ast_timings | 15.684 |
| describe_scenes | 7.616 |
| summarize_scenes | 9.656 |
| synthesize_synopsis | 10.660 |
| make_embedding | 2.536 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 24.870 |
| branch_yolo_total | 9.450 |
| branch_audio_total | 39.446 |
