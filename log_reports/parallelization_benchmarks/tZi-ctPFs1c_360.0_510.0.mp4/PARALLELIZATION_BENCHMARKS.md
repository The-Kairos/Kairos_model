# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 23:41:56 UTC | tZi-ctPFs1c_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 100.534 | 0.786 | 40.537 | 9.005 | 5.030 | 8.940 | 2.262 |

## 2026-06-26 23:41:56 UTC | tZi-ctPFs1c_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/tZi-ctPFs1c_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `100.534` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.786 |
| save_clips | - |
| sample_frames | 0.764 |
| caption_frames | 22.919 |
| sample_fps | 2.073 |
| detect_object_yolo | 6.820 |
| audio_scan | 12.708 |
| asr_timings | 11.612 |
| ast_timings | 16.209 |
| describe_scenes | 9.005 |
| summarize_scenes | 5.030 |
| synthesize_synopsis | 8.940 |
| make_embedding | 2.262 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 23.690 |
| branch_yolo_total | 8.899 |
| branch_audio_total | 40.537 |
