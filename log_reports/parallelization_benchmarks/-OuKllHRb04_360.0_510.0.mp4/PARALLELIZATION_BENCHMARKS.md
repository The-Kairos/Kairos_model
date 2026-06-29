# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 08:18:57 UTC | -OuKllHRb04_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 190.501 | 0.802 | 63.110 | 20.214 | 16.415 | 22.100 | 4.588 |

## 2026-06-24 08:18:57 UTC | -OuKllHRb04_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/-OuKllHRb04_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `190.501` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.802 |
| save_clips | - |
| sample_frames | 1.372 |
| caption_frames | 48.072 |
| sample_fps | 2.424 |
| detect_object_yolo | 10.045 |
| audio_scan | 15.942 |
| asr_timings | 9.987 |
| ast_timings | 37.173 |
| describe_scenes | 20.214 |
| summarize_scenes | 16.415 |
| synthesize_synopsis | 22.100 |
| make_embedding | 4.588 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 49.450 |
| branch_yolo_total | 12.475 |
| branch_audio_total | 63.110 |
