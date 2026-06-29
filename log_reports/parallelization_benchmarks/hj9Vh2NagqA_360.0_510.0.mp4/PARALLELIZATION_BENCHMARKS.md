# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 06:47:13 UTC | hj9Vh2NagqA_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 198.104 | 0.799 | 62.476 | 16.297 | 13.645 | 33.670 | 4.728 |

## 2026-06-26 06:47:13 UTC | hj9Vh2NagqA_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/hj9Vh2NagqA_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `198.104` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.799 |
| save_clips | - |
| sample_frames | 1.430 |
| caption_frames | 50.746 |
| sample_fps | 2.504 |
| detect_object_yolo | 10.390 |
| audio_scan | 15.168 |
| asr_timings | 8.812 |
| ast_timings | 38.487 |
| describe_scenes | 16.297 |
| summarize_scenes | 13.645 |
| synthesize_synopsis | 33.670 |
| make_embedding | 4.728 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 52.182 |
| branch_yolo_total | 12.900 |
| branch_audio_total | 62.476 |
