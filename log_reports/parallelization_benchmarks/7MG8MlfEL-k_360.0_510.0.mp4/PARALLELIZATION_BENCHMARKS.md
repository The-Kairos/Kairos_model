# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 13:52:19 UTC | 7MG8MlfEL-k_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 156.652 | 0.787 | 57.399 | 14.805 | 13.907 | 18.694 | 3.106 |

## 2026-06-24 13:52:19 UTC | 7MG8MlfEL-k_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/7MG8MlfEL-k_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `156.652` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.787 |
| save_clips | - |
| sample_frames | 0.970 |
| caption_frames | 34.694 |
| sample_fps | 2.232 |
| detect_object_yolo | 8.592 |
| audio_scan | 7.497 |
| asr_timings | 25.423 |
| ast_timings | 24.470 |
| describe_scenes | 14.805 |
| summarize_scenes | 13.907 |
| synthesize_synopsis | 18.694 |
| make_embedding | 3.106 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 35.670 |
| branch_yolo_total | 10.830 |
| branch_audio_total | 57.399 |
