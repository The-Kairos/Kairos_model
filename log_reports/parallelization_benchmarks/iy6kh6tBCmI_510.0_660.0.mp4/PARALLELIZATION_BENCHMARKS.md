# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 09:48:59 UTC | iy6kh6tBCmI_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 163.982 | 0.798 | 50.561 | 18.702 | 14.222 | 25.321 | 3.305 |

## 2026-06-26 09:48:59 UTC | iy6kh6tBCmI_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/iy6kh6tBCmI_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `163.982` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.798 |
| save_clips | - |
| sample_frames | 1.138 |
| caption_frames | 37.673 |
| sample_fps | 2.266 |
| detect_object_yolo | 8.549 |
| audio_scan | 13.009 |
| asr_timings | 10.235 |
| ast_timings | 27.308 |
| describe_scenes | 18.702 |
| summarize_scenes | 14.222 |
| synthesize_synopsis | 25.321 |
| make_embedding | 3.305 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 38.816 |
| branch_yolo_total | 10.821 |
| branch_audio_total | 50.561 |
