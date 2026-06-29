# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 14:10:00 UTC | 7Q3Gpf51QFU_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 221.305 | 0.832 | 85.556 | 21.621 | 13.498 | 27.470 | 4.702 |

## 2026-06-24 14:10:00 UTC | 7Q3Gpf51QFU_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/7Q3Gpf51QFU_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `221.305` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.832 |
| save_clips | - |
| sample_frames | 1.463 |
| caption_frames | 51.839 |
| sample_fps | 2.487 |
| detect_object_yolo | 10.443 |
| audio_scan | 15.895 |
| asr_timings | 31.521 |
| ast_timings | 38.131 |
| describe_scenes | 21.621 |
| summarize_scenes | 13.498 |
| synthesize_synopsis | 27.470 |
| make_embedding | 4.702 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 53.308 |
| branch_yolo_total | 12.936 |
| branch_audio_total | 85.556 |
