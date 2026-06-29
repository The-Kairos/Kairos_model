# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 11:45:20 UTC | OLVgoQEnvqg_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 240.114 | 0.791 | 72.568 | 28.086 | 26.963 | 16.975 | 6.523 |

## 2026-06-25 11:45:20 UTC | OLVgoQEnvqg_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/OLVgoQEnvqg_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `240.114` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.791 |
| save_clips | - |
| sample_frames | 1.943 |
| caption_frames | 69.225 |
| sample_fps | 2.691 |
| detect_object_yolo | 12.949 |
| audio_scan | 9.977 |
| asr_timings | 9.668 |
| ast_timings | 52.915 |
| describe_scenes | 28.086 |
| summarize_scenes | 26.963 |
| synthesize_synopsis | 16.975 |
| make_embedding | 6.523 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 71.173 |
| branch_yolo_total | 15.646 |
| branch_audio_total | 72.568 |
