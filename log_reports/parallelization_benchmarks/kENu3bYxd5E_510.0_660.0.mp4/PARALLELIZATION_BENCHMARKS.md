# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 13:34:42 UTC | kENu3bYxd5E_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 172.963 | 0.798 | 55.202 | 26.351 | 16.236 | 25.965 | 3.079 |

## 2026-06-26 13:34:42 UTC | kENu3bYxd5E_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/kENu3bYxd5E_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `172.963` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.798 |
| save_clips | - |
| sample_frames | 1.046 |
| caption_frames | 32.329 |
| sample_fps | 2.238 |
| detect_object_yolo | 8.314 |
| audio_scan | 14.030 |
| asr_timings | 16.763 |
| ast_timings | 24.401 |
| describe_scenes | 26.351 |
| summarize_scenes | 16.236 |
| synthesize_synopsis | 25.965 |
| make_embedding | 3.079 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 33.381 |
| branch_yolo_total | 10.558 |
| branch_audio_total | 55.202 |
