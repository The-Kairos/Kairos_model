# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 23:22:50 UTC | Dz0MY6ARnU4_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 134.466 | 0.667 | 72.299 | 7.655 | 7.975 | 7.061 | 2.479 |

## 2026-06-24 23:22:50 UTC | Dz0MY6ARnU4_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/Dz0MY6ARnU4_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `134.466` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.667 |
| save_clips | - |
| sample_frames | 0.589 |
| caption_frames | 24.777 |
| sample_fps | 1.892 |
| detect_object_yolo | 7.651 |
| audio_scan | 14.948 |
| asr_timings | 39.178 |
| ast_timings | 18.164 |
| describe_scenes | 7.655 |
| summarize_scenes | 7.975 |
| synthesize_synopsis | 7.061 |
| make_embedding | 2.479 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 25.371 |
| branch_yolo_total | 9.549 |
| branch_audio_total | 72.299 |
