# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 19:25:15 UTC | UqMooNqP7Hs_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 124.864 | 0.810 | 43.551 | 9.567 | 13.953 | 17.531 | 2.562 |

## 2026-06-25 19:25:15 UTC | UqMooNqP7Hs_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/UqMooNqP7Hs_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `124.864` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.810 |
| save_clips | - |
| sample_frames | 0.680 |
| caption_frames | 25.471 |
| sample_fps | 2.005 |
| detect_object_yolo | 7.339 |
| audio_scan | 14.923 |
| asr_timings | 9.690 |
| ast_timings | 18.930 |
| describe_scenes | 9.567 |
| summarize_scenes | 13.953 |
| synthesize_synopsis | 17.531 |
| make_embedding | 2.562 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 26.157 |
| branch_yolo_total | 9.351 |
| branch_audio_total | 43.551 |
