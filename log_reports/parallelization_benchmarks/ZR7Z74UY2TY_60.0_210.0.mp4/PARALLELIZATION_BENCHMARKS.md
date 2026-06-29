# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 22:28:15 UTC | ZR7Z74UY2TY_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 133.755 | 0.691 | 50.515 | 10.776 | 8.171 | 12.733 | 3.259 |

## 2026-06-25 22:28:15 UTC | ZR7Z74UY2TY_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/ZR7Z74UY2TY_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `133.755` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.691 |
| save_clips | - |
| sample_frames | 1.049 |
| caption_frames | 34.982 |
| sample_fps | 2.068 |
| detect_object_yolo | 8.103 |
| audio_scan | 12.736 |
| asr_timings | 9.913 |
| ast_timings | 27.858 |
| describe_scenes | 10.776 |
| summarize_scenes | 8.171 |
| synthesize_synopsis | 12.733 |
| make_embedding | 3.259 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 36.037 |
| branch_yolo_total | 10.177 |
| branch_audio_total | 50.515 |
