# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 12:54:23 UTC | k4LLzwmwJS8_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 193.836 | 0.801 | 58.788 | 22.822 | 24.377 | 17.413 | 4.497 |

## 2026-06-26 12:54:23 UTC | k4LLzwmwJS8_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/k4LLzwmwJS8_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `193.836` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.801 |
| save_clips | - |
| sample_frames | 1.227 |
| caption_frames | 49.907 |
| sample_fps | 2.405 |
| detect_object_yolo | 10.156 |
| audio_scan | 10.785 |
| asr_timings | 9.202 |
| ast_timings | 38.793 |
| describe_scenes | 22.822 |
| summarize_scenes | 24.377 |
| synthesize_synopsis | 17.413 |
| make_embedding | 4.497 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 51.141 |
| branch_yolo_total | 12.567 |
| branch_audio_total | 58.788 |
