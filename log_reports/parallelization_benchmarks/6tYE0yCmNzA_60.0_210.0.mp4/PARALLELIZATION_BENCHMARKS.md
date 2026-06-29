# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 13:03:10 UTC | 6tYE0yCmNzA_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 147.635 | 0.675 | 51.710 | 13.750 | 11.547 | 22.017 | 3.161 |

## 2026-06-24 13:03:10 UTC | 6tYE0yCmNzA_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/6tYE0yCmNzA_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `147.635` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.675 |
| save_clips | - |
| sample_frames | 0.790 |
| caption_frames | 32.704 |
| sample_fps | 1.966 |
| detect_object_yolo | 7.937 |
| audio_scan | 15.997 |
| asr_timings | 11.468 |
| ast_timings | 24.236 |
| describe_scenes | 13.750 |
| summarize_scenes | 11.547 |
| synthesize_synopsis | 22.017 |
| make_embedding | 3.161 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 33.500 |
| branch_yolo_total | 9.909 |
| branch_audio_total | 51.710 |
