# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 19:59:15 UTC | VayyLoioSAk_660.0_810.0.mp4 | sequential | gemini | gemini-embedding-001 | 187.549 | 0.621 | 62.534 | 18.498 | 15.397 | 20.861 | 4.463 |

## 2026-06-25 19:59:15 UTC | VayyLoioSAk_660.0_810.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/VayyLoioSAk_660.0_810.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `187.549` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.621 |
| save_clips | - |
| sample_frames | 1.117 |
| caption_frames | 49.711 |
| sample_fps | 2.175 |
| detect_object_yolo | 10.684 |
| audio_scan | 14.078 |
| asr_timings | 10.694 |
| ast_timings | 37.753 |
| describe_scenes | 18.498 |
| summarize_scenes | 15.397 |
| synthesize_synopsis | 20.861 |
| make_embedding | 4.463 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 50.834 |
| branch_yolo_total | 12.865 |
| branch_audio_total | 62.534 |
