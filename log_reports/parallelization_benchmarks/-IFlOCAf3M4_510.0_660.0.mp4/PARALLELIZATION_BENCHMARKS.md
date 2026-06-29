# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 07:55:36 UTC | -IFlOCAf3M4_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 1693.910 | 0.824 | 1580.310 | 19.785 | 27.180 | 20.161 | 3.088 |

## 2026-06-24 07:55:36 UTC | -IFlOCAf3M4_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/-IFlOCAf3M4_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `1693.910` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.824 |
| save_clips | - |
| sample_frames | 0.971 |
| caption_frames | 30.019 |
| sample_fps | 2.117 |
| detect_object_yolo | 8.109 |
| audio_scan | 13.724 |
| asr_timings | 1542.983 |
| ast_timings | 23.595 |
| describe_scenes | 19.785 |
| summarize_scenes | 27.180 |
| synthesize_synopsis | 20.161 |
| make_embedding | 3.088 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 30.996 |
| branch_yolo_total | 10.232 |
| branch_audio_total | 1580.310 |
