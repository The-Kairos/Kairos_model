# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 10:22:53 UTC | jAVGP5XwXns_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 126.954 | 0.660 | 37.115 | 10.571 | 29.348 | 24.974 | 1.824 |

## 2026-06-26 10:22:53 UTC | jAVGP5XwXns_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/jAVGP5XwXns_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `126.954` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.660 |
| save_clips | - |
| sample_frames | 0.331 |
| caption_frames | 13.328 |
| sample_fps | 1.828 |
| detect_object_yolo | 5.578 |
| audio_scan | 16.111 |
| asr_timings | 11.070 |
| ast_timings | 9.925 |
| describe_scenes | 10.571 |
| summarize_scenes | 29.348 |
| synthesize_synopsis | 24.974 |
| make_embedding | 1.824 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 13.665 |
| branch_yolo_total | 7.412 |
| branch_audio_total | 37.115 |
