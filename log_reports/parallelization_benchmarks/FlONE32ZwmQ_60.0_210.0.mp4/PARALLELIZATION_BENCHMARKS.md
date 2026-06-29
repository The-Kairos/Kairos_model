# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 00:29:03 UTC | FlONE32ZwmQ_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 72.247 | 0.625 | 29.837 | 5.357 | 4.162 | 9.157 | 1.713 |

## 2026-06-25 00:29:03 UTC | FlONE32ZwmQ_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/FlONE32ZwmQ_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `72.247` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.625 |
| save_clips | - |
| sample_frames | 0.228 |
| caption_frames | 11.939 |
| sample_fps | 1.671 |
| detect_object_yolo | 6.082 |
| audio_scan | 8.643 |
| asr_timings | 13.518 |
| ast_timings | 7.667 |
| describe_scenes | 5.357 |
| summarize_scenes | 4.162 |
| synthesize_synopsis | 9.157 |
| make_embedding | 1.713 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 12.173 |
| branch_yolo_total | 7.759 |
| branch_audio_total | 29.837 |
