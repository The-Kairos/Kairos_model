# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 05:49:01 UTC | K4ReSUwx6iQ_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 191.974 | 0.779 | 75.914 | 17.487 | 21.079 | 10.230 | 3.851 |

## 2026-06-25 05:49:01 UTC | K4ReSUwx6iQ_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/K4ReSUwx6iQ_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `191.974` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.779 |
| save_clips | - |
| sample_frames | 1.343 |
| caption_frames | 47.830 |
| sample_fps | 2.331 |
| detect_object_yolo | 9.741 |
| audio_scan | 8.570 |
| asr_timings | 35.023 |
| ast_timings | 32.312 |
| describe_scenes | 17.487 |
| summarize_scenes | 21.079 |
| synthesize_synopsis | 10.230 |
| make_embedding | 3.851 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 49.179 |
| branch_yolo_total | 12.077 |
| branch_audio_total | 75.914 |
