# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 05:10:34 UTC | JRkGew32t9E_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 137.980 | 0.673 | 45.710 | 10.256 | 19.002 | 14.735 | 2.994 |

## 2026-06-25 05:10:34 UTC | JRkGew32t9E_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/JRkGew32t9E_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `137.980` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.673 |
| save_clips | - |
| sample_frames | 0.772 |
| caption_frames | 32.105 |
| sample_fps | 1.979 |
| detect_object_yolo | 8.299 |
| audio_scan | 11.821 |
| asr_timings | 10.643 |
| ast_timings | 23.237 |
| describe_scenes | 10.256 |
| summarize_scenes | 19.002 |
| synthesize_synopsis | 14.735 |
| make_embedding | 2.994 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 32.883 |
| branch_yolo_total | 10.284 |
| branch_audio_total | 45.710 |
