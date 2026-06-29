# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 10:47:41 UTC | jL7v3S2mFww_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 133.630 | 0.743 | 46.359 | 12.898 | 11.085 | 19.223 | 2.780 |

## 2026-06-26 10:47:41 UTC | jL7v3S2mFww_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/jL7v3S2mFww_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `133.630` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.743 |
| save_clips | - |
| sample_frames | 0.910 |
| caption_frames | 28.602 |
| sample_fps | 2.059 |
| detect_object_yolo | 7.573 |
| audio_scan | 15.104 |
| asr_timings | 10.347 |
| ast_timings | 20.900 |
| describe_scenes | 12.898 |
| summarize_scenes | 11.085 |
| synthesize_synopsis | 19.223 |
| make_embedding | 2.780 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 29.518 |
| branch_yolo_total | 9.638 |
| branch_audio_total | 46.359 |
