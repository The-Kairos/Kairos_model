# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 13:21:24 UTC | kBAWeVHNlBo_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 252.700 | 0.799 | 76.991 | 23.569 | 35.343 | 22.754 | 6.267 |

## 2026-06-26 13:21:24 UTC | kBAWeVHNlBo_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/kBAWeVHNlBo_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `252.700` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.799 |
| save_clips | - |
| sample_frames | 1.574 |
| caption_frames | 68.384 |
| sample_fps | 2.631 |
| detect_object_yolo | 12.930 |
| audio_scan | 16.279 |
| asr_timings | 11.524 |
| ast_timings | 49.180 |
| describe_scenes | 23.569 |
| summarize_scenes | 35.343 |
| synthesize_synopsis | 22.754 |
| make_embedding | 6.267 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 69.964 |
| branch_yolo_total | 15.567 |
| branch_audio_total | 76.991 |
