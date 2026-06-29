# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 13:27:28 UTC | 0HAACVba7kI_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 128.185 | 0.844 | 55.576 | 7.987 | 9.534 | 9.057 | 2.818 |

## 2026-06-27 13:27:28 UTC | 0HAACVba7kI_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/0HAACVba7kI_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `128.185` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.844 |
| save_clips | - |
| sample_frames | 0.850 |
| caption_frames | 29.993 |
| sample_fps | 2.143 |
| detect_object_yolo | 7.973 |
| audio_scan | 14.855 |
| asr_timings | 19.260 |
| ast_timings | 21.453 |
| describe_scenes | 7.987 |
| summarize_scenes | 9.534 |
| synthesize_synopsis | 9.057 |
| make_embedding | 2.818 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 30.849 |
| branch_yolo_total | 10.121 |
| branch_audio_total | 55.576 |
