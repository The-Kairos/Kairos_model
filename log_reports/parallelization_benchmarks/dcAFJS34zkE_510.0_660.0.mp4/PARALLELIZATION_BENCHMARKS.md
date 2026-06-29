# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 03:08:17 UTC | dcAFJS34zkE_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 150.903 | 0.779 | 52.698 | 11.215 | 8.228 | 11.273 | 4.533 |

## 2026-06-26 03:08:17 UTC | dcAFJS34zkE_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/dcAFJS34zkE_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `150.903` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.779 |
| save_clips | - |
| sample_frames | 1.288 |
| caption_frames | 47.667 |
| sample_fps | 2.400 |
| detect_object_yolo | 9.425 |
| audio_scan | 9.778 |
| asr_timings | 6.570 |
| ast_timings | 36.342 |
| describe_scenes | 11.215 |
| summarize_scenes | 8.228 |
| synthesize_synopsis | 11.273 |
| make_embedding | 4.533 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 48.962 |
| branch_yolo_total | 11.830 |
| branch_audio_total | 52.698 |
