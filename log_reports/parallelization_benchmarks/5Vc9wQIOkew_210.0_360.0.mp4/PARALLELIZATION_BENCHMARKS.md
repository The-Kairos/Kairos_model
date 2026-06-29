# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 11:54:41 UTC | 5Vc9wQIOkew_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 163.998 | 0.817 | 48.150 | 13.057 | 29.595 | 19.599 | 3.314 |

## 2026-06-24 11:54:41 UTC | 5Vc9wQIOkew_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/5Vc9wQIOkew_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `163.998` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.817 |
| save_clips | - |
| sample_frames | 1.352 |
| caption_frames | 36.037 |
| sample_fps | 2.320 |
| detect_object_yolo | 8.359 |
| audio_scan | 12.819 |
| asr_timings | 8.498 |
| ast_timings | 26.824 |
| describe_scenes | 13.057 |
| summarize_scenes | 29.595 |
| synthesize_synopsis | 19.599 |
| make_embedding | 3.314 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 37.395 |
| branch_yolo_total | 10.685 |
| branch_audio_total | 48.150 |
