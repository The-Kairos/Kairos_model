# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 03:37:52 UTC | xDt8qY4uhtk_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 156.794 | 0.764 | 61.567 | 12.052 | 11.490 | 5.669 | 4.167 |

## 2026-06-27 03:37:52 UTC | xDt8qY4uhtk_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/xDt8qY4uhtk_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `156.794` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.764 |
| save_clips | - |
| sample_frames | 1.125 |
| caption_frames | 46.630 |
| sample_fps | 2.293 |
| detect_object_yolo | 9.651 |
| audio_scan | 14.126 |
| asr_timings | 12.464 |
| ast_timings | 34.968 |
| describe_scenes | 12.052 |
| summarize_scenes | 11.490 |
| synthesize_synopsis | 5.669 |
| make_embedding | 4.167 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 47.760 |
| branch_yolo_total | 11.950 |
| branch_audio_total | 61.567 |
