# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 16:53:06 UTC | nwgLQy1b_Ro_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 141.297 | 0.824 | 52.897 | 10.793 | 10.191 | 9.675 | 3.647 |

## 2026-06-27 16:53:06 UTC | nwgLQy1b_Ro_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/nwgLQy1b_Ro_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `141.297` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.824 |
| save_clips | - |
| sample_frames | 1.441 |
| caption_frames | 39.514 |
| sample_fps | 2.407 |
| detect_object_yolo | 8.523 |
| audio_scan | 14.893 |
| asr_timings | 8.870 |
| ast_timings | 29.126 |
| describe_scenes | 10.793 |
| summarize_scenes | 10.191 |
| synthesize_synopsis | 9.675 |
| make_embedding | 3.647 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 40.961 |
| branch_yolo_total | 10.936 |
| branch_audio_total | 52.897 |
