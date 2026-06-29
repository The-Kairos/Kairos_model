# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 16:30:23 UTC | 7gVqrrCbcOw_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 142.814 | 0.788 | 51.224 | 10.713 | 7.471 | 18.164 | 3.406 |

## 2026-06-24 16:30:23 UTC | 7gVqrrCbcOw_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/7gVqrrCbcOw_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `142.814` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.788 |
| save_clips | - |
| sample_frames | 0.995 |
| caption_frames | 38.177 |
| sample_fps | 2.196 |
| detect_object_yolo | 8.300 |
| audio_scan | 14.896 |
| asr_timings | 9.599 |
| ast_timings | 26.721 |
| describe_scenes | 10.713 |
| summarize_scenes | 7.471 |
| synthesize_synopsis | 18.164 |
| make_embedding | 3.406 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 39.178 |
| branch_yolo_total | 10.502 |
| branch_audio_total | 51.224 |
