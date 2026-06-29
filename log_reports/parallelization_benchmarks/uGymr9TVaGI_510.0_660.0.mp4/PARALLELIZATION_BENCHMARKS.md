# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 00:51:00 UTC | uGymr9TVaGI_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 66.193 | 0.776 | 31.651 | 4.249 | 2.674 | 8.140 | 1.289 |

## 2026-06-27 00:51:00 UTC | uGymr9TVaGI_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/uGymr9TVaGI_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `66.193` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.776 |
| save_clips | - |
| sample_frames | 0.093 |
| caption_frames | 8.399 |
| sample_fps | 1.766 |
| detect_object_yolo | 5.705 |
| audio_scan | 16.022 |
| asr_timings | 10.305 |
| ast_timings | 5.316 |
| describe_scenes | 4.249 |
| summarize_scenes | 2.674 |
| synthesize_synopsis | 8.140 |
| make_embedding | 1.289 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 8.498 |
| branch_yolo_total | 7.477 |
| branch_audio_total | 31.651 |
