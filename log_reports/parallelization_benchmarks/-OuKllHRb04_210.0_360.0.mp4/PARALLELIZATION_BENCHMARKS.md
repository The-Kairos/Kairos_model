# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 08:15:45 UTC | -OuKllHRb04_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 212.762 | 0.833 | 64.837 | 21.419 | 30.130 | 20.237 | 5.222 |

## 2026-06-24 08:15:45 UTC | -OuKllHRb04_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/-OuKllHRb04_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `212.762` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.833 |
| save_clips | - |
| sample_frames | 1.437 |
| caption_frames | 54.328 |
| sample_fps | 2.445 |
| detect_object_yolo | 10.536 |
| audio_scan | 12.807 |
| asr_timings | 10.664 |
| ast_timings | 41.358 |
| describe_scenes | 21.419 |
| summarize_scenes | 30.130 |
| synthesize_synopsis | 20.237 |
| make_embedding | 5.222 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 55.771 |
| branch_yolo_total | 12.987 |
| branch_audio_total | 64.837 |
