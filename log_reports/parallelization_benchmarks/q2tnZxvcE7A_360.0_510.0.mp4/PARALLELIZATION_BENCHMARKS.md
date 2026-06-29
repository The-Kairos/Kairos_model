# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-28 08:25:53 UTC | q2tnZxvcE7A_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 184.714 | 0.731 | 96.624 | 10.541 | 9.516 | 6.259 | 3.819 |

## 2026-06-28 08:25:53 UTC | q2tnZxvcE7A_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/q2tnZxvcE7A_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `184.714` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.731 |
| save_clips | - |
| sample_frames | 1.570 |
| caption_frames | 42.942 |
| sample_fps | 2.287 |
| detect_object_yolo | 9.013 |
| audio_scan | 14.782 |
| asr_timings | 48.743 |
| ast_timings | 33.090 |
| describe_scenes | 10.541 |
| summarize_scenes | 9.516 |
| synthesize_synopsis | 6.259 |
| make_embedding | 3.819 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 44.518 |
| branch_yolo_total | 11.305 |
| branch_audio_total | 96.624 |
