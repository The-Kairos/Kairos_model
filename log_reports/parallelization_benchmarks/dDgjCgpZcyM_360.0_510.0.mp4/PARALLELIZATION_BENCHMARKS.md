# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 02:48:53 UTC | dDgjCgpZcyM_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 160.286 | 0.646 | 57.993 | 14.962 | 10.101 | 13.984 | 3.873 |

## 2026-06-26 02:48:53 UTC | dDgjCgpZcyM_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/dDgjCgpZcyM_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `160.286` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.646 |
| save_clips | - |
| sample_frames | 1.294 |
| caption_frames | 44.635 |
| sample_fps | 2.216 |
| detect_object_yolo | 9.178 |
| audio_scan | 6.544 |
| asr_timings | 18.235 |
| ast_timings | 33.205 |
| describe_scenes | 14.962 |
| summarize_scenes | 10.101 |
| synthesize_synopsis | 13.984 |
| make_embedding | 3.873 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 45.935 |
| branch_yolo_total | 11.400 |
| branch_audio_total | 57.993 |
