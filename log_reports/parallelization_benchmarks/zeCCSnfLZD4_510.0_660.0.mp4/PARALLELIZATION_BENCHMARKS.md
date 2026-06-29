# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 05:45:59 UTC | zeCCSnfLZD4_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 119.713 | 0.640 | 43.063 | 8.668 | 8.329 | 9.286 | 3.267 |

## 2026-06-27 05:45:59 UTC | zeCCSnfLZD4_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/zeCCSnfLZD4_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `119.713` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.640 |
| save_clips | - |
| sample_frames | 0.745 |
| caption_frames | 34.886 |
| sample_fps | 1.842 |
| detect_object_yolo | 7.586 |
| audio_scan | 6.753 |
| asr_timings | 10.101 |
| ast_timings | 26.201 |
| describe_scenes | 8.668 |
| summarize_scenes | 8.329 |
| synthesize_synopsis | 9.286 |
| make_embedding | 3.267 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 35.636 |
| branch_yolo_total | 9.434 |
| branch_audio_total | 43.063 |
