# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 07:33:10 UTC | i-BqG0x21io_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 278.004 | 0.811 | 84.918 | 30.675 | 25.405 | 35.089 | 6.621 |

## 2026-06-26 07:33:10 UTC | i-BqG0x21io_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/i-BqG0x21io_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `278.004` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.811 |
| save_clips | - |
| sample_frames | 1.793 |
| caption_frames | 75.098 |
| sample_fps | 2.803 |
| detect_object_yolo | 13.368 |
| audio_scan | 14.020 |
| asr_timings | 15.724 |
| ast_timings | 55.165 |
| describe_scenes | 30.675 |
| summarize_scenes | 25.405 |
| synthesize_synopsis | 35.089 |
| make_embedding | 6.621 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 76.897 |
| branch_yolo_total | 16.177 |
| branch_audio_total | 84.918 |
