# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 18:16:26 UTC | 9erWlhsParM_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 89.459 | 0.625 | 32.660 | 6.614 | 11.497 | 17.853 | 1.502 |

## 2026-06-24 18:16:26 UTC | 9erWlhsParM_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/9erWlhsParM_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `89.459` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.625 |
| save_clips | - |
| sample_frames | 0.128 |
| caption_frames | 9.805 |
| sample_fps | 1.602 |
| detect_object_yolo | 5.789 |
| audio_scan | 14.895 |
| asr_timings | 10.402 |
| ast_timings | 7.354 |
| describe_scenes | 6.614 |
| summarize_scenes | 11.497 |
| synthesize_synopsis | 17.853 |
| make_embedding | 1.502 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 9.940 |
| branch_yolo_total | 7.397 |
| branch_audio_total | 32.660 |
