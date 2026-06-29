# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 11:44:43 UTC | jna4ylDjww0_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 170.531 | 0.699 | 53.323 | 23.011 | 12.643 | 27.028 | 3.335 |

## 2026-06-26 11:44:43 UTC | jna4ylDjww0_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/jna4ylDjww0_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `170.531` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.699 |
| save_clips | - |
| sample_frames | 0.847 |
| caption_frames | 37.389 |
| sample_fps | 2.079 |
| detect_object_yolo | 8.731 |
| audio_scan | 16.168 |
| asr_timings | 9.928 |
| ast_timings | 27.218 |
| describe_scenes | 23.011 |
| summarize_scenes | 12.643 |
| synthesize_synopsis | 27.028 |
| make_embedding | 3.335 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 38.242 |
| branch_yolo_total | 10.816 |
| branch_audio_total | 53.323 |
