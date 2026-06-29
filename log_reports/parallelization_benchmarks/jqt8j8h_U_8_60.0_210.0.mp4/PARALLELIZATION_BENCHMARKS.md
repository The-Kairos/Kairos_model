# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 12:19:39 UTC | jqt8j8h_U_8_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 160.332 | 0.693 | 49.843 | 20.735 | 12.570 | 16.297 | 3.851 |

## 2026-06-26 12:19:39 UTC | jqt8j8h_U_8_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/jqt8j8h_U_8_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `160.332` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.693 |
| save_clips | - |
| sample_frames | 1.215 |
| caption_frames | 42.242 |
| sample_fps | 2.212 |
| detect_object_yolo | 9.281 |
| audio_scan | 6.544 |
| asr_timings | 10.369 |
| ast_timings | 32.921 |
| describe_scenes | 20.735 |
| summarize_scenes | 12.570 |
| synthesize_synopsis | 16.297 |
| make_embedding | 3.851 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 43.463 |
| branch_yolo_total | 11.498 |
| branch_audio_total | 49.843 |
