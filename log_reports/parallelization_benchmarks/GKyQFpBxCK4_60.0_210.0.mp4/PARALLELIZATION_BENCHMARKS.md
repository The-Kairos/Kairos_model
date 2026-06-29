# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 01:44:08 UTC | GKyQFpBxCK4_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 163.850 | 0.654 | 55.662 | 9.397 | 13.362 | 26.912 | 3.576 |

## 2026-06-25 01:44:08 UTC | GKyQFpBxCK4_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/GKyQFpBxCK4_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `163.850` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.654 |
| save_clips | - |
| sample_frames | 1.049 |
| caption_frames | 40.974 |
| sample_fps | 2.024 |
| detect_object_yolo | 8.839 |
| audio_scan | 14.925 |
| asr_timings | 10.538 |
| ast_timings | 30.190 |
| describe_scenes | 9.397 |
| summarize_scenes | 13.362 |
| synthesize_synopsis | 26.912 |
| make_embedding | 3.576 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 42.029 |
| branch_yolo_total | 10.869 |
| branch_audio_total | 55.662 |
