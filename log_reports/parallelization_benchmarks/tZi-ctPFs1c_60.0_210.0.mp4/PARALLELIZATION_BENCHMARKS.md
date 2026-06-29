# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 23:44:07 UTC | tZi-ctPFs1c_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 129.488 | 0.777 | 51.871 | 9.681 | 5.738 | 8.145 | 3.336 |

## 2026-06-26 23:44:07 UTC | tZi-ctPFs1c_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/tZi-ctPFs1c_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `129.488` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.777 |
| save_clips | - |
| sample_frames | 1.032 |
| caption_frames | 36.744 |
| sample_fps | 2.222 |
| detect_object_yolo | 8.518 |
| audio_scan | 15.054 |
| asr_timings | 9.021 |
| ast_timings | 27.788 |
| describe_scenes | 9.681 |
| summarize_scenes | 5.738 |
| synthesize_synopsis | 8.145 |
| make_embedding | 3.336 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 37.782 |
| branch_yolo_total | 10.746 |
| branch_audio_total | 51.871 |
