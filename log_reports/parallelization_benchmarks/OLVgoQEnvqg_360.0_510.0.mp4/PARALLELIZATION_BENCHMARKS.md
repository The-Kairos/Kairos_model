# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 11:41:19 UTC | OLVgoQEnvqg_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 207.352 | 0.820 | 65.572 | 24.140 | 26.157 | 20.973 | 4.556 |

## 2026-06-25 11:41:19 UTC | OLVgoQEnvqg_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/OLVgoQEnvqg_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `207.352` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.820 |
| save_clips | - |
| sample_frames | 1.813 |
| caption_frames | 49.230 |
| sample_fps | 2.645 |
| detect_object_yolo | 10.042 |
| audio_scan | 14.487 |
| asr_timings | 12.682 |
| ast_timings | 38.394 |
| describe_scenes | 24.140 |
| summarize_scenes | 26.157 |
| synthesize_synopsis | 20.973 |
| make_embedding | 4.556 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 51.048 |
| branch_yolo_total | 12.693 |
| branch_audio_total | 65.572 |
