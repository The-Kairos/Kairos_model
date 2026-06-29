# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 12:16:12 UTC | 6MTL7aBxgR0_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 131.571 | 0.779 | 40.177 | 13.834 | 12.812 | 17.706 | 3.535 |

## 2026-06-24 12:16:12 UTC | 6MTL7aBxgR0_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/6MTL7aBxgR0_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `131.571` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.779 |
| save_clips | - |
| sample_frames | 0.925 |
| caption_frames | 39.245 |
| sample_fps | 2.198 |
| detect_object_yolo | 8.283 |
| audio_scan | 3.818 |
| asr_timings | 0.000 |
| ast_timings | 27.022 |
| describe_scenes | 13.834 |
| summarize_scenes | 12.812 |
| synthesize_synopsis | 17.706 |
| make_embedding | 3.535 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 40.177 |
| branch_yolo_total | 10.487 |
| branch_audio_total | 30.849 |
