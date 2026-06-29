# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 03:17:28 UTC | HJiDesesiWg_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 156.947 | 0.703 | 56.651 | 17.539 | 8.023 | 11.345 | 3.902 |

## 2026-06-25 03:17:28 UTC | HJiDesesiWg_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/HJiDesesiWg_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `156.947` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.703 |
| save_clips | - |
| sample_frames | 1.328 |
| caption_frames | 43.946 |
| sample_fps | 2.291 |
| detect_object_yolo | 9.825 |
| audio_scan | 14.992 |
| asr_timings | 9.266 |
| ast_timings | 32.384 |
| describe_scenes | 17.539 |
| summarize_scenes | 8.023 |
| synthesize_synopsis | 11.345 |
| make_embedding | 3.902 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 45.279 |
| branch_yolo_total | 12.121 |
| branch_audio_total | 56.651 |
