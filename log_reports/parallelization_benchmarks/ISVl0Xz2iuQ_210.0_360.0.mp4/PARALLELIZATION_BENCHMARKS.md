# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 04:48:27 UTC | ISVl0Xz2iuQ_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 193.196 | 0.692 | 58.365 | 22.130 | 16.192 | 21.612 | 5.201 |

## 2026-06-25 04:48:27 UTC | ISVl0Xz2iuQ_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/ISVl0Xz2iuQ_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `193.196` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.692 |
| save_clips | - |
| sample_frames | 1.348 |
| caption_frames | 53.263 |
| sample_fps | 2.334 |
| detect_object_yolo | 10.660 |
| audio_scan | 8.568 |
| asr_timings | 8.468 |
| ast_timings | 41.321 |
| describe_scenes | 22.130 |
| summarize_scenes | 16.192 |
| synthesize_synopsis | 21.612 |
| make_embedding | 5.201 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 54.616 |
| branch_yolo_total | 13.000 |
| branch_audio_total | 58.365 |
