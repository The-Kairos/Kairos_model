# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 14:11:54 UTC | PV8maMvPqhw_660.0_810.0.mp4 | sequential | gemini | gemini-embedding-001 | 257.557 | 0.643 | 68.453 | 39.135 | 50.028 | 23.994 | 5.160 |

## 2026-06-25 14:11:54 UTC | PV8maMvPqhw_660.0_810.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/PV8maMvPqhw_660.0_810.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `257.557` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.643 |
| save_clips | - |
| sample_frames | 1.728 |
| caption_frames | 53.532 |
| sample_fps | 2.437 |
| detect_object_yolo | 10.917 |
| audio_scan | 15.810 |
| asr_timings | 11.116 |
| ast_timings | 41.519 |
| describe_scenes | 39.135 |
| summarize_scenes | 50.028 |
| synthesize_synopsis | 23.994 |
| make_embedding | 5.160 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 55.266 |
| branch_yolo_total | 13.360 |
| branch_audio_total | 68.453 |
