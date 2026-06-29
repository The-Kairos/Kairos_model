# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 02:44:16 UTC | dBY9k9W7k-I_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 124.297 | 0.785 | 43.875 | 12.461 | 9.545 | 10.363 | 3.027 |

## 2026-06-26 02:44:16 UTC | dBY9k9W7k-I_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/dBY9k9W7k-I_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `124.297` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.785 |
| save_clips | - |
| sample_frames | 1.050 |
| caption_frames | 31.778 |
| sample_fps | 2.218 |
| detect_object_yolo | 7.801 |
| audio_scan | 9.764 |
| asr_timings | 9.552 |
| ast_timings | 24.551 |
| describe_scenes | 12.461 |
| summarize_scenes | 9.545 |
| synthesize_synopsis | 10.363 |
| make_embedding | 3.027 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 32.834 |
| branch_yolo_total | 10.025 |
| branch_audio_total | 43.875 |
