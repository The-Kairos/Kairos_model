# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 02:58:57 UTC | wQoQU0blZgw_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 90.042 | 0.778 | 33.336 | 4.976 | 6.311 | 13.010 | 2.024 |

## 2026-06-27 02:58:57 UTC | wQoQU0blZgw_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/wQoQU0blZgw_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `90.042` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.778 |
| save_clips | - |
| sample_frames | 0.421 |
| caption_frames | 19.094 |
| sample_fps | 1.934 |
| detect_object_yolo | 6.736 |
| audio_scan | 11.900 |
| asr_timings | 8.408 |
| ast_timings | 13.019 |
| describe_scenes | 4.976 |
| summarize_scenes | 6.311 |
| synthesize_synopsis | 13.010 |
| make_embedding | 2.024 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 19.521 |
| branch_yolo_total | 8.676 |
| branch_audio_total | 33.336 |
