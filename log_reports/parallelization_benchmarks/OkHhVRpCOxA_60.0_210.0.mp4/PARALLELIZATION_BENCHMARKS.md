# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 12:24:38 UTC | OkHhVRpCOxA_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 282.525 | 0.795 | 106.055 | 31.674 | 22.644 | 37.469 | 5.654 |

## 2026-06-25 12:24:38 UTC | OkHhVRpCOxA_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/OkHhVRpCOxA_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `282.525` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.795 |
| save_clips | - |
| sample_frames | 1.582 |
| caption_frames | 60.816 |
| sample_fps | 2.607 |
| detect_object_yolo | 11.813 |
| audio_scan | 15.436 |
| asr_timings | 44.374 |
| ast_timings | 46.237 |
| describe_scenes | 31.674 |
| summarize_scenes | 22.644 |
| synthesize_synopsis | 37.469 |
| make_embedding | 5.654 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 62.405 |
| branch_yolo_total | 14.426 |
| branch_audio_total | 106.055 |
