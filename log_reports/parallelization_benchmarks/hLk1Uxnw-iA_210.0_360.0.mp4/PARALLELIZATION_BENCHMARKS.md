# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 06:13:03 UTC | hLk1Uxnw-iA_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 232.518 | 0.649 | 87.086 | 15.777 | 44.053 | 22.362 | 3.951 |

## 2026-06-26 06:13:03 UTC | hLk1Uxnw-iA_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/hLk1Uxnw-iA_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `232.518` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.649 |
| save_clips | - |
| sample_frames | 1.178 |
| caption_frames | 44.121 |
| sample_fps | 2.145 |
| detect_object_yolo | 9.767 |
| audio_scan | 10.737 |
| asr_timings | 42.920 |
| ast_timings | 33.421 |
| describe_scenes | 15.777 |
| summarize_scenes | 44.053 |
| synthesize_synopsis | 22.362 |
| make_embedding | 3.951 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 45.305 |
| branch_yolo_total | 11.918 |
| branch_audio_total | 87.086 |
