# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 13:51:56 UTC | PPSICA2UeP0_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 300.082 | 0.797 | 68.379 | 35.535 | 60.275 | 55.294 | 5.123 |

## 2026-06-25 13:51:56 UTC | PPSICA2UeP0_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/PPSICA2UeP0_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `300.082` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.797 |
| save_clips | - |
| sample_frames | 1.503 |
| caption_frames | 57.976 |
| sample_fps | 2.547 |
| detect_object_yolo | 11.191 |
| audio_scan | 16.794 |
| asr_timings | 10.549 |
| ast_timings | 41.028 |
| describe_scenes | 35.535 |
| summarize_scenes | 60.275 |
| synthesize_synopsis | 55.294 |
| make_embedding | 5.123 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 59.485 |
| branch_yolo_total | 13.744 |
| branch_audio_total | 68.379 |
