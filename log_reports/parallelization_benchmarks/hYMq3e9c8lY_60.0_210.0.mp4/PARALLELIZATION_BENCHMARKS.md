# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 06:26:06 UTC | hYMq3e9c8lY_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 203.513 | 0.644 | 63.535 | 16.901 | 26.594 | 18.095 | 5.337 |

## 2026-06-26 06:26:06 UTC | hYMq3e9c8lY_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/hYMq3e9c8lY_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `203.513` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.644 |
| save_clips | - |
| sample_frames | 1.382 |
| caption_frames | 56.513 |
| sample_fps | 2.286 |
| detect_object_yolo | 10.796 |
| audio_scan | 11.850 |
| asr_timings | 10.237 |
| ast_timings | 41.432 |
| describe_scenes | 16.901 |
| summarize_scenes | 26.594 |
| synthesize_synopsis | 18.095 |
| make_embedding | 5.337 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 57.901 |
| branch_yolo_total | 13.087 |
| branch_audio_total | 63.535 |
