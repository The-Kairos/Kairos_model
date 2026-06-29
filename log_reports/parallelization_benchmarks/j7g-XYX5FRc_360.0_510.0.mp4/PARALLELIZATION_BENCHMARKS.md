# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 10:11:44 UTC | j7g-XYX5FRc_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 222.470 | 0.821 | 71.172 | 26.232 | 16.916 | 21.369 | 5.839 |

## 2026-06-26 10:11:44 UTC | j7g-XYX5FRc_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/j7g-XYX5FRc_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `222.470` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.821 |
| save_clips | - |
| sample_frames | 1.511 |
| caption_frames | 62.544 |
| sample_fps | 2.588 |
| detect_object_yolo | 12.057 |
| audio_scan | 14.023 |
| asr_timings | 9.836 |
| ast_timings | 47.305 |
| describe_scenes | 26.232 |
| summarize_scenes | 16.916 |
| synthesize_synopsis | 21.369 |
| make_embedding | 5.839 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 64.060 |
| branch_yolo_total | 14.651 |
| branch_audio_total | 71.172 |
