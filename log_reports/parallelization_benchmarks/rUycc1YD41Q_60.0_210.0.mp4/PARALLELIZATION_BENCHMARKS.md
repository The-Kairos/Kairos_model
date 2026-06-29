# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 18:41:39 UTC | rUycc1YD41Q_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 160.120 | 0.778 | 53.114 | 15.621 | 20.710 | 11.655 | 3.565 |

## 2026-06-26 18:41:39 UTC | rUycc1YD41Q_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/rUycc1YD41Q_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `160.120` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.778 |
| save_clips | - |
| sample_frames | 1.112 |
| caption_frames | 40.482 |
| sample_fps | 2.260 |
| detect_object_yolo | 9.384 |
| audio_scan | 11.810 |
| asr_timings | 10.981 |
| ast_timings | 30.314 |
| describe_scenes | 15.621 |
| summarize_scenes | 20.710 |
| synthesize_synopsis | 11.655 |
| make_embedding | 3.565 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 41.600 |
| branch_yolo_total | 11.650 |
| branch_audio_total | 53.114 |
