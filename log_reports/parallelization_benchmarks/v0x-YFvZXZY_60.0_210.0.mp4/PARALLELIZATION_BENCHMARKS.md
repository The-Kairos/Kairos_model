# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 02:02:58 UTC | v0x-YFvZXZY_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 157.752 | 0.822 | 61.832 | 10.654 | 8.194 | 8.819 | 4.564 |

## 2026-06-27 02:02:58 UTC | v0x-YFvZXZY_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/v0x-YFvZXZY_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `157.752` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.822 |
| save_clips | - |
| sample_frames | 1.415 |
| caption_frames | 47.892 |
| sample_fps | 2.527 |
| detect_object_yolo | 9.628 |
| audio_scan | 15.067 |
| asr_timings | 11.317 |
| ast_timings | 35.439 |
| describe_scenes | 10.654 |
| summarize_scenes | 8.194 |
| synthesize_synopsis | 8.819 |
| make_embedding | 4.564 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 49.313 |
| branch_yolo_total | 12.160 |
| branch_audio_total | 61.832 |
