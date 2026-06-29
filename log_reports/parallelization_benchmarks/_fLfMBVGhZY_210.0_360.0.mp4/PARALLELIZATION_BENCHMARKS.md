# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 23:59:24 UTC | _fLfMBVGhZY_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 134.070 | 0.779 | 49.772 | 7.797 | 6.858 | 10.581 | 3.569 |

## 2026-06-25 23:59:24 UTC | _fLfMBVGhZY_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/_fLfMBVGhZY_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `134.070` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.779 |
| save_clips | - |
| sample_frames | 1.053 |
| caption_frames | 40.881 |
| sample_fps | 2.254 |
| detect_object_yolo | 9.093 |
| audio_scan | 9.596 |
| asr_timings | 9.489 |
| ast_timings | 30.679 |
| describe_scenes | 7.797 |
| summarize_scenes | 6.858 |
| synthesize_synopsis | 10.581 |
| make_embedding | 3.569 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 41.940 |
| branch_yolo_total | 11.353 |
| branch_audio_total | 49.772 |
