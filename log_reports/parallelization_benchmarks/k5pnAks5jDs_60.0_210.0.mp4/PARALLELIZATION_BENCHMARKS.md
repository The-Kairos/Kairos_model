# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 13:12:45 UTC | k5pnAks5jDs_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 210.243 | 0.838 | 73.506 | 26.407 | 21.722 | 21.419 | 4.145 |

## 2026-06-26 13:12:45 UTC | k5pnAks5jDs_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/k5pnAks5jDs_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `210.243` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.838 |
| save_clips | - |
| sample_frames | 1.280 |
| caption_frames | 46.735 |
| sample_fps | 2.405 |
| detect_object_yolo | 10.277 |
| audio_scan | 13.173 |
| asr_timings | 24.644 |
| ast_timings | 35.679 |
| describe_scenes | 26.407 |
| summarize_scenes | 21.722 |
| synthesize_synopsis | 21.419 |
| make_embedding | 4.145 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 48.020 |
| branch_yolo_total | 12.688 |
| branch_audio_total | 73.506 |
