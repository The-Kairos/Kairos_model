# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 21:33:32 UTC | XJlrte91c4A_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 134.798 | 0.790 | 50.715 | 12.209 | 10.106 | 12.676 | 3.029 |

## 2026-06-25 21:33:32 UTC | XJlrte91c4A_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/XJlrte91c4A_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `134.798` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.790 |
| save_clips | - |
| sample_frames | 0.764 |
| caption_frames | 32.851 |
| sample_fps | 2.145 |
| detect_object_yolo | 8.097 |
| audio_scan | 16.243 |
| asr_timings | 10.022 |
| ast_timings | 24.441 |
| describe_scenes | 12.209 |
| summarize_scenes | 10.106 |
| synthesize_synopsis | 12.676 |
| make_embedding | 3.029 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 33.621 |
| branch_yolo_total | 10.247 |
| branch_audio_total | 50.715 |
