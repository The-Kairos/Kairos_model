# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 19:36:36 UTC | s92vs3XgNPE_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 134.183 | 0.799 | 51.187 | 11.784 | 11.653 | 11.948 | 3.075 |

## 2026-06-26 19:36:36 UTC | s92vs3XgNPE_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/s92vs3XgNPE_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `134.183` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.799 |
| save_clips | - |
| sample_frames | 0.947 |
| caption_frames | 31.165 |
| sample_fps | 2.140 |
| detect_object_yolo | 8.038 |
| audio_scan | 10.730 |
| asr_timings | 15.994 |
| ast_timings | 24.454 |
| describe_scenes | 11.784 |
| summarize_scenes | 11.653 |
| synthesize_synopsis | 11.948 |
| make_embedding | 3.075 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 32.119 |
| branch_yolo_total | 10.184 |
| branch_audio_total | 51.187 |
