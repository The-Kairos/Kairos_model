# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 23:58:54 UTC | EtXnXxOegko_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 98.533 | 0.786 | 37.950 | 5.854 | 5.311 | 9.344 | 2.506 |

## 2026-06-24 23:58:54 UTC | EtXnXxOegko_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/EtXnXxOegko_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `98.533` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.786 |
| save_clips | - |
| sample_frames | 0.684 |
| caption_frames | 25.694 |
| sample_fps | 2.007 |
| detect_object_yolo | 6.951 |
| audio_scan | 8.609 |
| asr_timings | 10.345 |
| ast_timings | 18.988 |
| describe_scenes | 5.854 |
| summarize_scenes | 5.311 |
| synthesize_synopsis | 9.344 |
| make_embedding | 2.506 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 26.384 |
| branch_yolo_total | 8.964 |
| branch_audio_total | 37.950 |
