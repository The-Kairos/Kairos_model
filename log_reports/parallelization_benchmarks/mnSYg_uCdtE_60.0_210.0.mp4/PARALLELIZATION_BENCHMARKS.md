# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 16:13:51 UTC | mnSYg_uCdtE_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 174.406 | 0.803 | 64.808 | 14.075 | 8.013 | 6.359 | 5.342 |

## 2026-06-27 16:13:51 UTC | mnSYg_uCdtE_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/mnSYg_uCdtE_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `174.406` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.803 |
| save_clips | - |
| sample_frames | 1.767 |
| caption_frames | 57.663 |
| sample_fps | 2.617 |
| detect_object_yolo | 11.563 |
| audio_scan | 11.705 |
| asr_timings | 9.496 |
| ast_timings | 43.599 |
| describe_scenes | 14.075 |
| summarize_scenes | 8.013 |
| synthesize_synopsis | 6.359 |
| make_embedding | 5.342 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 59.436 |
| branch_yolo_total | 14.186 |
| branch_audio_total | 64.808 |
