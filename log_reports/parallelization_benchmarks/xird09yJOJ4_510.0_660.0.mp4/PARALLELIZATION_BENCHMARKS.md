# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 04:00:01 UTC | xird09yJOJ4_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 127.721 | 0.770 | 50.180 | 6.488 | 7.274 | 9.743 | 3.299 |

## 2026-06-27 04:00:01 UTC | xird09yJOJ4_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/xird09yJOJ4_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `127.721` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.770 |
| save_clips | - |
| sample_frames | 0.925 |
| caption_frames | 36.236 |
| sample_fps | 2.211 |
| detect_object_yolo | 9.187 |
| audio_scan | 13.970 |
| asr_timings | 9.678 |
| ast_timings | 26.524 |
| describe_scenes | 6.488 |
| summarize_scenes | 7.274 |
| synthesize_synopsis | 9.743 |
| make_embedding | 3.299 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 37.167 |
| branch_yolo_total | 11.403 |
| branch_audio_total | 50.180 |
