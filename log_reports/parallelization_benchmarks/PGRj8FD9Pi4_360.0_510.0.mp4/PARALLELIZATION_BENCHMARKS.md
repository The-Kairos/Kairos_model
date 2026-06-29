# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 13:01:09 UTC | PGRj8FD9Pi4_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 227.241 | 0.642 | 66.400 | 33.999 | 25.296 | 22.437 | 5.713 |

## 2026-06-25 13:01:09 UTC | PGRj8FD9Pi4_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/PGRj8FD9Pi4_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `227.241` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.642 |
| save_clips | - |
| sample_frames | 1.331 |
| caption_frames | 56.851 |
| sample_fps | 2.283 |
| detect_object_yolo | 10.894 |
| audio_scan | 11.032 |
| asr_timings | 11.498 |
| ast_timings | 43.861 |
| describe_scenes | 33.999 |
| summarize_scenes | 25.296 |
| synthesize_synopsis | 22.437 |
| make_embedding | 5.713 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 58.188 |
| branch_yolo_total | 13.184 |
| branch_audio_total | 66.400 |
