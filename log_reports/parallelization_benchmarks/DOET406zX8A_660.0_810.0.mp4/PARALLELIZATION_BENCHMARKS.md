# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 22:42:01 UTC | DOET406zX8A_660.0_810.0.mp4 | sequential | gemini | gemini-embedding-001 | 94.474 | 0.630 | 29.674 | 7.878 | 6.250 | 13.275 | 2.847 |

## 2026-06-24 22:42:01 UTC | DOET406zX8A_660.0_810.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/DOET406zX8A_660.0_810.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `94.474` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.630 |
| save_clips | - |
| sample_frames | 0.679 |
| caption_frames | 28.990 |
| sample_fps | 1.897 |
| detect_object_yolo | 7.704 |
| audio_scan | 3.912 |
| asr_timings | 0.000 |
| ast_timings | 18.967 |
| describe_scenes | 7.878 |
| summarize_scenes | 6.250 |
| synthesize_synopsis | 13.275 |
| make_embedding | 2.847 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 29.674 |
| branch_yolo_total | 9.607 |
| branch_audio_total | 22.888 |
