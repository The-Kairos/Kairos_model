# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 19:17:59 UTC | UYp5bX4rgOQ_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 173.700 | 0.836 | 64.416 | 20.377 | 11.427 | 11.743 | 4.094 |

## 2026-06-25 19:17:59 UTC | UYp5bX4rgOQ_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/UYp5bX4rgOQ_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `173.700` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.836 |
| save_clips | - |
| sample_frames | 1.327 |
| caption_frames | 45.759 |
| sample_fps | 2.451 |
| detect_object_yolo | 9.848 |
| audio_scan | 16.040 |
| asr_timings | 13.640 |
| ast_timings | 34.728 |
| describe_scenes | 20.377 |
| summarize_scenes | 11.427 |
| synthesize_synopsis | 11.743 |
| make_embedding | 4.094 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 47.092 |
| branch_yolo_total | 12.304 |
| branch_audio_total | 64.416 |
