# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 13:37:03 UTC | 7JUPCETmSR8_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 253.890 | 0.731 | 65.433 | 29.518 | 61.358 | 21.190 | 5.183 |

## 2026-06-24 13:37:03 UTC | 7JUPCETmSR8_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/7JUPCETmSR8_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `253.890` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.731 |
| save_clips | - |
| sample_frames | 1.632 |
| caption_frames | 54.320 |
| sample_fps | 2.485 |
| detect_object_yolo | 10.582 |
| audio_scan | 16.129 |
| asr_timings | 8.333 |
| ast_timings | 40.962 |
| describe_scenes | 29.518 |
| summarize_scenes | 61.358 |
| synthesize_synopsis | 21.190 |
| make_embedding | 5.183 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 55.959 |
| branch_yolo_total | 13.073 |
| branch_audio_total | 65.433 |
