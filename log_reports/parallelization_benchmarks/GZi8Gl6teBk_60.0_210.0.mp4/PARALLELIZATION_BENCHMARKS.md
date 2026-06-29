# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 01:56:48 UTC | GZi8Gl6teBk_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 128.126 | 0.665 | 47.428 | 9.826 | 10.079 | 13.115 | 3.011 |

## 2026-06-25 01:56:48 UTC | GZi8Gl6teBk_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/GZi8Gl6teBk_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `128.126` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.665 |
| save_clips | - |
| sample_frames | 1.051 |
| caption_frames | 31.647 |
| sample_fps | 2.043 |
| detect_object_yolo | 7.867 |
| audio_scan | 14.838 |
| asr_timings | 8.689 |
| ast_timings | 23.893 |
| describe_scenes | 9.826 |
| summarize_scenes | 10.079 |
| synthesize_synopsis | 13.115 |
| make_embedding | 3.011 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 32.704 |
| branch_yolo_total | 9.916 |
| branch_audio_total | 47.428 |
