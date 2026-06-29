# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 13:09:02 UTC | PGRj8FD9Pi4_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 229.574 | 0.651 | 63.838 | 34.009 | 27.475 | 24.435 | 6.026 |

## 2026-06-25 13:09:02 UTC | PGRj8FD9Pi4_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/PGRj8FD9Pi4_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `229.574` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.651 |
| save_clips | - |
| sample_frames | 1.402 |
| caption_frames | 57.172 |
| sample_fps | 2.315 |
| detect_object_yolo | 10.835 |
| audio_scan | 7.721 |
| asr_timings | 12.383 |
| ast_timings | 43.726 |
| describe_scenes | 34.009 |
| summarize_scenes | 27.475 |
| synthesize_synopsis | 24.435 |
| make_embedding | 6.026 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 58.580 |
| branch_yolo_total | 13.155 |
| branch_audio_total | 63.838 |
