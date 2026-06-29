# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 06:19:30 UTC | hLk1Uxnw-iA_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 190.231 | 0.679 | 80.373 | 13.324 | 24.139 | 22.591 | 3.860 |

## 2026-06-26 06:19:30 UTC | hLk1Uxnw-iA_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/hLk1Uxnw-iA_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `190.231` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.679 |
| save_clips | - |
| sample_frames | 0.915 |
| caption_frames | 32.431 |
| sample_fps | 2.022 |
| detect_object_yolo | 8.431 |
| audio_scan | 11.984 |
| asr_timings | 44.015 |
| ast_timings | 24.358 |
| describe_scenes | 13.324 |
| summarize_scenes | 24.139 |
| synthesize_synopsis | 22.591 |
| make_embedding | 3.860 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 33.352 |
| branch_yolo_total | 10.459 |
| branch_audio_total | 80.373 |
