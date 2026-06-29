# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 08:25:37 UTC | -SqS_5tSv78_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 180.585 | 0.837 | 55.308 | 26.973 | 18.671 | 17.659 | 3.977 |

## 2026-06-24 08:25:37 UTC | -SqS_5tSv78_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/-SqS_5tSv78_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `180.585` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.837 |
| save_clips | - |
| sample_frames | 1.279 |
| caption_frames | 43.025 |
| sample_fps | 2.347 |
| detect_object_yolo | 9.122 |
| audio_scan | 12.796 |
| asr_timings | 9.648 |
| ast_timings | 32.848 |
| describe_scenes | 26.973 |
| summarize_scenes | 18.671 |
| synthesize_synopsis | 17.659 |
| make_embedding | 3.977 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 44.311 |
| branch_yolo_total | 11.475 |
| branch_audio_total | 55.308 |
