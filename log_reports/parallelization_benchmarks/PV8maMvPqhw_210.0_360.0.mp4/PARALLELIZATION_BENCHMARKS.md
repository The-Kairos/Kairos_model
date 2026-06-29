# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 13:55:57 UTC | PV8maMvPqhw_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 239.789 | 0.678 | 61.955 | 35.280 | 44.645 | 24.661 | 4.724 |

## 2026-06-25 13:55:57 UTC | PV8maMvPqhw_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/PV8maMvPqhw_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `239.789` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.678 |
| save_clips | - |
| sample_frames | 1.754 |
| caption_frames | 51.612 |
| sample_fps | 2.437 |
| detect_object_yolo | 10.600 |
| audio_scan | 14.321 |
| asr_timings | 9.624 |
| ast_timings | 38.001 |
| describe_scenes | 35.280 |
| summarize_scenes | 44.645 |
| synthesize_synopsis | 24.661 |
| make_embedding | 4.724 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 53.372 |
| branch_yolo_total | 13.042 |
| branch_audio_total | 61.955 |
