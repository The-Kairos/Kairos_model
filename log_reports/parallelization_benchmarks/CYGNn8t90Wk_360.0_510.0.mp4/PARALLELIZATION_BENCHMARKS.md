# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 20:35:52 UTC | CYGNn8t90Wk_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 180.543 | 0.629 | 65.064 | 13.552 | 8.456 | 19.881 | 5.040 |

## 2026-06-24 20:35:52 UTC | CYGNn8t90Wk_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/CYGNn8t90Wk_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `180.543` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.629 |
| save_clips | - |
| sample_frames | 1.203 |
| caption_frames | 52.888 |
| sample_fps | 2.212 |
| detect_object_yolo | 10.219 |
| audio_scan | 12.801 |
| asr_timings | 11.125 |
| ast_timings | 41.130 |
| describe_scenes | 13.552 |
| summarize_scenes | 8.456 |
| synthesize_synopsis | 19.881 |
| make_embedding | 5.040 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 54.096 |
| branch_yolo_total | 12.436 |
| branch_audio_total | 65.064 |
