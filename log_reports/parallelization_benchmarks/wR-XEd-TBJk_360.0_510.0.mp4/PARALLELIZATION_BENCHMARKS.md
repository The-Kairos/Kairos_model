# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 03:06:38 UTC | wR-XEd-TBJk_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 143.507 | 0.776 | 57.231 | 8.490 | 11.827 | 7.395 | 3.577 |

## 2026-06-27 03:06:38 UTC | wR-XEd-TBJk_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/wR-XEd-TBJk_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `143.507` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.776 |
| save_clips | - |
| sample_frames | 1.000 |
| caption_frames | 40.624 |
| sample_fps | 2.218 |
| detect_object_yolo | 8.963 |
| audio_scan | 15.145 |
| asr_timings | 12.090 |
| ast_timings | 29.988 |
| describe_scenes | 8.490 |
| summarize_scenes | 11.827 |
| synthesize_synopsis | 7.395 |
| make_embedding | 3.577 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 41.630 |
| branch_yolo_total | 11.186 |
| branch_audio_total | 57.231 |
