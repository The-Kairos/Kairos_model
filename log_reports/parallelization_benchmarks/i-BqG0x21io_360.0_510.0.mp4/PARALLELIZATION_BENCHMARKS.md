# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 07:37:13 UTC | i-BqG0x21io_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 242.117 | 0.820 | 83.185 | 26.021 | 24.581 | 24.189 | 5.466 |

## 2026-06-26 07:37:13 UTC | i-BqG0x21io_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/i-BqG0x21io_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `242.117` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.820 |
| save_clips | - |
| sample_frames | 1.565 |
| caption_frames | 61.124 |
| sample_fps | 2.619 |
| detect_object_yolo | 11.133 |
| audio_scan | 13.993 |
| asr_timings | 25.341 |
| ast_timings | 43.843 |
| describe_scenes | 26.021 |
| summarize_scenes | 24.581 |
| synthesize_synopsis | 24.189 |
| make_embedding | 5.466 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 62.695 |
| branch_yolo_total | 13.758 |
| branch_audio_total | 83.185 |
