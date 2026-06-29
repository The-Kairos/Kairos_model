# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 03:35:14 UTC | x6QkZM27EVw_660.0_810.0.mp4 | sequential | gemini | gemini-embedding-001 | 169.234 | 0.656 | 92.342 | 11.760 | 4.376 | 7.487 | 3.267 |

## 2026-06-27 03:35:14 UTC | x6QkZM27EVw_660.0_810.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/x6QkZM27EVw_660.0_810.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `169.234` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.656 |
| save_clips | - |
| sample_frames | 1.260 |
| caption_frames | 35.965 |
| sample_fps | 2.208 |
| detect_object_yolo | 8.496 |
| audio_scan | 11.875 |
| asr_timings | 53.271 |
| ast_timings | 27.189 |
| describe_scenes | 11.760 |
| summarize_scenes | 4.376 |
| synthesize_synopsis | 7.487 |
| make_embedding | 3.267 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 37.231 |
| branch_yolo_total | 10.710 |
| branch_audio_total | 92.342 |
