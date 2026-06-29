# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 01:55:39 UTC | c1lX1gd5I78_660.0_810.0.mp4 | sequential | gemini | gemini-embedding-001 | 163.549 | 0.846 | 58.280 | 14.652 | 8.682 | 11.042 | 4.115 |

## 2026-06-26 01:55:39 UTC | c1lX1gd5I78_660.0_810.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/c1lX1gd5I78_660.0_810.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `163.549` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.846 |
| save_clips | - |
| sample_frames | 1.696 |
| caption_frames | 50.305 |
| sample_fps | 2.600 |
| detect_object_yolo | 9.873 |
| audio_scan | 12.067 |
| asr_timings | 9.532 |
| ast_timings | 36.672 |
| describe_scenes | 14.652 |
| summarize_scenes | 8.682 |
| synthesize_synopsis | 11.042 |
| make_embedding | 4.115 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 52.008 |
| branch_yolo_total | 12.478 |
| branch_audio_total | 58.280 |
