# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 13:02:33 UTC | k5pnAks5jDs_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 248.999 | 0.789 | 64.035 | 27.772 | 59.877 | 25.792 | 4.824 |

## 2026-06-26 13:02:33 UTC | k5pnAks5jDs_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/k5pnAks5jDs_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `248.999` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.789 |
| save_clips | - |
| sample_frames | 1.383 |
| caption_frames | 49.949 |
| sample_fps | 2.459 |
| detect_object_yolo | 10.687 |
| audio_scan | 14.086 |
| asr_timings | 10.749 |
| ast_timings | 39.191 |
| describe_scenes | 27.772 |
| summarize_scenes | 59.877 |
| synthesize_synopsis | 25.792 |
| make_embedding | 4.824 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 51.338 |
| branch_yolo_total | 13.152 |
| branch_audio_total | 64.035 |
