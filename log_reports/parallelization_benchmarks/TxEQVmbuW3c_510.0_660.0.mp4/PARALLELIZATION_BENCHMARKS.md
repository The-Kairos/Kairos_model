# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 18:23:10 UTC | TxEQVmbuW3c_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 165.718 | 0.809 | 59.447 | 10.930 | 11.056 | 20.500 | 4.107 |

## 2026-06-25 18:23:10 UTC | TxEQVmbuW3c_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/TxEQVmbuW3c_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `165.718` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.809 |
| save_clips | - |
| sample_frames | 1.235 |
| caption_frames | 44.320 |
| sample_fps | 2.347 |
| detect_object_yolo | 9.580 |
| audio_scan | 14.951 |
| asr_timings | 9.389 |
| ast_timings | 35.099 |
| describe_scenes | 10.930 |
| summarize_scenes | 11.056 |
| synthesize_synopsis | 20.500 |
| make_embedding | 4.107 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 45.561 |
| branch_yolo_total | 11.933 |
| branch_audio_total | 59.447 |
