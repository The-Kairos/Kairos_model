# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 23:07:50 UTC | DtLH2de0Wwc_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 146.150 | 0.801 | 80.290 | 6.570 | 7.633 | 12.152 | 2.523 |

## 2026-06-24 23:07:50 UTC | DtLH2de0Wwc_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/DtLH2de0Wwc_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `146.150` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.801 |
| save_clips | - |
| sample_frames | 0.702 |
| caption_frames | 24.722 |
| sample_fps | 2.087 |
| detect_object_yolo | 7.201 |
| audio_scan | 11.960 |
| asr_timings | 49.658 |
| ast_timings | 18.663 |
| describe_scenes | 6.570 |
| summarize_scenes | 7.633 |
| synthesize_synopsis | 12.152 |
| make_embedding | 2.523 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 25.430 |
| branch_yolo_total | 9.293 |
| branch_audio_total | 80.290 |
