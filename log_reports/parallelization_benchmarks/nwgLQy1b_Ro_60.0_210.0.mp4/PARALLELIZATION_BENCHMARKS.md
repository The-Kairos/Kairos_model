# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 16:55:41 UTC | nwgLQy1b_Ro_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 154.055 | 0.801 | 58.514 | 12.525 | 9.482 | 8.727 | 4.145 |

## 2026-06-27 16:55:41 UTC | nwgLQy1b_Ro_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/nwgLQy1b_Ro_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `154.055` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.801 |
| save_clips | - |
| sample_frames | 1.696 |
| caption_frames | 44.746 |
| sample_fps | 2.503 |
| detect_object_yolo | 9.513 |
| audio_scan | 14.857 |
| asr_timings | 8.950 |
| ast_timings | 34.698 |
| describe_scenes | 12.525 |
| summarize_scenes | 9.482 |
| synthesize_synopsis | 8.727 |
| make_embedding | 4.145 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 46.448 |
| branch_yolo_total | 12.021 |
| branch_audio_total | 58.514 |
