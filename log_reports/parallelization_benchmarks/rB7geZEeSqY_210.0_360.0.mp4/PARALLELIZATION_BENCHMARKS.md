# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 18:29:07 UTC | rB7geZEeSqY_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 121.960 | 0.786 | 37.651 | 12.199 | 12.876 | 17.592 | 2.555 |

## 2026-06-26 18:29:07 UTC | rB7geZEeSqY_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/rB7geZEeSqY_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `121.960` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.786 |
| save_clips | - |
| sample_frames | 0.609 |
| caption_frames | 26.758 |
| sample_fps | 1.980 |
| detect_object_yolo | 7.558 |
| audio_scan | 10.411 |
| asr_timings | 8.594 |
| ast_timings | 18.638 |
| describe_scenes | 12.199 |
| summarize_scenes | 12.876 |
| synthesize_synopsis | 17.592 |
| make_embedding | 2.555 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 27.373 |
| branch_yolo_total | 9.543 |
| branch_audio_total | 37.651 |
