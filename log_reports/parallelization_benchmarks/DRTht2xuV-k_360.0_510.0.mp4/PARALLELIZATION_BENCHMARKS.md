# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 22:48:05 UTC | DRTht2xuV-k_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 162.133 | 0.787 | 66.853 | 13.359 | 14.208 | 9.631 | 3.546 |

## 2026-06-24 22:48:05 UTC | DRTht2xuV-k_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/DRTht2xuV-k_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `162.133` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.787 |
| save_clips | - |
| sample_frames | 1.127 |
| caption_frames | 39.826 |
| sample_fps | 2.305 |
| detect_object_yolo | 9.048 |
| audio_scan | 15.083 |
| asr_timings | 21.421 |
| ast_timings | 30.341 |
| describe_scenes | 13.359 |
| summarize_scenes | 14.208 |
| synthesize_synopsis | 9.631 |
| make_embedding | 3.546 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 40.959 |
| branch_yolo_total | 11.358 |
| branch_audio_total | 66.853 |
