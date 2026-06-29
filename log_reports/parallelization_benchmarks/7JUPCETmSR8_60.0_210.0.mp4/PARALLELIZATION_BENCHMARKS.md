# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 13:46:34 UTC | 7JUPCETmSR8_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 161.525 | 0.675 | 58.112 | 14.740 | 13.139 | 22.199 | 3.288 |

## 2026-06-24 13:46:34 UTC | 7JUPCETmSR8_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/7JUPCETmSR8_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `161.525` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.675 |
| save_clips | - |
| sample_frames | 0.920 |
| caption_frames | 36.582 |
| sample_fps | 2.026 |
| detect_object_yolo | 8.444 |
| audio_scan | 14.773 |
| asr_timings | 16.351 |
| ast_timings | 26.979 |
| describe_scenes | 14.740 |
| summarize_scenes | 13.139 |
| synthesize_synopsis | 22.199 |
| make_embedding | 3.288 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 37.507 |
| branch_yolo_total | 10.476 |
| branch_audio_total | 58.112 |
