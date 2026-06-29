# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 21:10:42 UTC | WlJGA2-wZQ4_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 214.614 | 0.786 | 97.188 | 11.674 | 23.127 | 8.620 | 4.663 |

## 2026-06-25 21:10:42 UTC | WlJGA2-wZQ4_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/WlJGA2-wZQ4_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `214.614` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.786 |
| save_clips | - |
| sample_frames | 1.423 |
| caption_frames | 52.405 |
| sample_fps | 2.464 |
| detect_object_yolo | 10.810 |
| audio_scan | 14.072 |
| asr_timings | 43.711 |
| ast_timings | 39.397 |
| describe_scenes | 11.674 |
| summarize_scenes | 23.127 |
| synthesize_synopsis | 8.620 |
| make_embedding | 4.663 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 53.834 |
| branch_yolo_total | 13.281 |
| branch_audio_total | 97.188 |
