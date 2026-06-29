# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 17:00:38 UTC | o0fdleH293A_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 138.589 | 0.771 | 51.846 | 10.741 | 12.411 | 10.261 | 3.421 |

## 2026-06-27 17:00:38 UTC | o0fdleH293A_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/o0fdleH293A_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `138.589` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.771 |
| save_clips | - |
| sample_frames | 0.852 |
| caption_frames | 36.429 |
| sample_fps | 2.158 |
| detect_object_yolo | 8.313 |
| audio_scan | 13.737 |
| asr_timings | 11.894 |
| ast_timings | 26.207 |
| describe_scenes | 10.741 |
| summarize_scenes | 12.411 |
| synthesize_synopsis | 10.261 |
| make_embedding | 3.421 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 37.287 |
| branch_yolo_total | 10.478 |
| branch_audio_total | 51.846 |
