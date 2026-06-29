# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 20:32:50 UTC | CYGNn8t90Wk_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 196.064 | 0.631 | 66.840 | 16.750 | 15.706 | 9.984 | 5.771 |

## 2026-06-24 20:32:50 UTC | CYGNn8t90Wk_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/CYGNn8t90Wk_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `196.064` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.631 |
| save_clips | - |
| sample_frames | 1.459 |
| caption_frames | 63.081 |
| sample_fps | 2.387 |
| detect_object_yolo | 12.045 |
| audio_scan | 9.747 |
| asr_timings | 9.947 |
| ast_timings | 47.137 |
| describe_scenes | 16.750 |
| summarize_scenes | 15.706 |
| synthesize_synopsis | 9.984 |
| make_embedding | 5.771 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 64.546 |
| branch_yolo_total | 14.438 |
| branch_audio_total | 66.840 |
