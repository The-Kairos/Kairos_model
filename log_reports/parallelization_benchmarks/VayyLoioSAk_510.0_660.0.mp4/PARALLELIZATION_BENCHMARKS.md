# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 19:53:00 UTC | VayyLoioSAk_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 210.555 | 0.640 | 71.279 | 26.910 | 14.115 | 12.587 | 5.751 |

## 2026-06-25 19:53:00 UTC | VayyLoioSAk_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/VayyLoioSAk_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `210.555` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.640 |
| save_clips | - |
| sample_frames | 1.499 |
| caption_frames | 62.232 |
| sample_fps | 2.337 |
| detect_object_yolo | 11.801 |
| audio_scan | 14.850 |
| asr_timings | 10.591 |
| ast_timings | 45.829 |
| describe_scenes | 26.910 |
| summarize_scenes | 14.115 |
| synthesize_synopsis | 12.587 |
| make_embedding | 5.751 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 63.737 |
| branch_yolo_total | 14.144 |
| branch_audio_total | 71.279 |
