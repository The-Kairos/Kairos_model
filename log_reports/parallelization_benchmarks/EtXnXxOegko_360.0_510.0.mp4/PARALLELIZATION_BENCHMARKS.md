# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 23:57:14 UTC | EtXnXxOegko_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 100.920 | 0.816 | 42.118 | 5.650 | 4.830 | 11.024 | 2.256 |

## 2026-06-24 23:57:14 UTC | EtXnXxOegko_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/EtXnXxOegko_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `100.920` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.816 |
| save_clips | - |
| sample_frames | 0.586 |
| caption_frames | 22.965 |
| sample_fps | 2.015 |
| detect_object_yolo | 7.183 |
| audio_scan | 15.139 |
| asr_timings | 10.811 |
| ast_timings | 16.160 |
| describe_scenes | 5.650 |
| summarize_scenes | 4.830 |
| synthesize_synopsis | 11.024 |
| make_embedding | 2.256 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 23.557 |
| branch_yolo_total | 9.204 |
| branch_audio_total | 42.118 |
