# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 06:33:04 UTC | zvEaxc9WD14_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 179.152 | 0.781 | 61.478 | 17.333 | 12.237 | 8.855 | 5.475 |

## 2026-06-27 06:33:04 UTC | zvEaxc9WD14_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/zvEaxc9WD14_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `179.152` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.781 |
| save_clips | - |
| sample_frames | 1.549 |
| caption_frames | 56.489 |
| sample_fps | 2.584 |
| detect_object_yolo | 10.956 |
| audio_scan | 8.526 |
| asr_timings | 8.639 |
| ast_timings | 44.305 |
| describe_scenes | 17.333 |
| summarize_scenes | 12.237 |
| synthesize_synopsis | 8.855 |
| make_embedding | 5.475 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 58.044 |
| branch_yolo_total | 13.546 |
| branch_audio_total | 61.478 |
