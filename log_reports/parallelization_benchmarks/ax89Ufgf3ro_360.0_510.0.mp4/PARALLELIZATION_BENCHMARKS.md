# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 00:30:00 UTC | ax89Ufgf3ro_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 126.393 | 0.651 | 54.067 | 10.622 | 6.550 | 10.128 | 2.751 |

## 2026-06-26 00:30:00 UTC | ax89Ufgf3ro_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/ax89Ufgf3ro_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `126.393` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.651 |
| save_clips | - |
| sample_frames | 0.903 |
| caption_frames | 29.589 |
| sample_fps | 1.952 |
| detect_object_yolo | 7.667 |
| audio_scan | 12.932 |
| asr_timings | 18.977 |
| ast_timings | 22.149 |
| describe_scenes | 10.622 |
| summarize_scenes | 6.550 |
| synthesize_synopsis | 10.128 |
| make_embedding | 2.751 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 30.497 |
| branch_yolo_total | 9.625 |
| branch_audio_total | 54.067 |
