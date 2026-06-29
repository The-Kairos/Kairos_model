# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-21 22:25:03 UTC | 3r7kOP_nYNo_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 178.604 | 0.694 | 61.959 | 11.981 | 17.361 | 8.438 | 5.393 |

## 2026-06-21 22:25:03 UTC | 3r7kOP_nYNo_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/3r7kOP_nYNo_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `178.604` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.694 |
| save_clips | - |
| sample_frames | 1.614 |
| caption_frames | 56.284 |
| sample_fps | 2.473 |
| detect_object_yolo | 10.985 |
| audio_scan | 11.822 |
| asr_timings | 6.392 |
| ast_timings | 43.736 |
| describe_scenes | 11.981 |
| summarize_scenes | 17.361 |
| synthesize_synopsis | 8.438 |
| make_embedding | 5.393 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 57.903 |
| branch_yolo_total | 13.464 |
| branch_audio_total | 61.959 |
