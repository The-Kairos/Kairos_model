# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 13:43:51 UTC | 7JUPCETmSR8_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 211.495 | 0.705 | 52.620 | 19.751 | 60.414 | 21.244 | 3.684 |

## 2026-06-24 13:43:51 UTC | 7JUPCETmSR8_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/7JUPCETmSR8_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `211.495` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.705 |
| save_clips | - |
| sample_frames | 1.220 |
| caption_frames | 39.574 |
| sample_fps | 2.158 |
| detect_object_yolo | 8.705 |
| audio_scan | 12.807 |
| asr_timings | 9.838 |
| ast_timings | 29.966 |
| describe_scenes | 19.751 |
| summarize_scenes | 60.414 |
| synthesize_synopsis | 21.244 |
| make_embedding | 3.684 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 40.800 |
| branch_yolo_total | 10.869 |
| branch_audio_total | 52.620 |
