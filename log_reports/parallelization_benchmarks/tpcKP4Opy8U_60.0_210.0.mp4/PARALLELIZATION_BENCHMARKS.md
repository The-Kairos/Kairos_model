# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 00:00:02 UTC | tpcKP4Opy8U_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 150.252 | 0.623 | 57.488 | 11.330 | 10.686 | 8.980 | 3.839 |

## 2026-06-27 00:00:02 UTC | tpcKP4Opy8U_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/tpcKP4Opy8U_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `150.252` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.623 |
| save_clips | - |
| sample_frames | 1.289 |
| caption_frames | 43.158 |
| sample_fps | 2.128 |
| detect_object_yolo | 9.299 |
| audio_scan | 15.951 |
| asr_timings | 8.306 |
| ast_timings | 33.223 |
| describe_scenes | 11.330 |
| summarize_scenes | 10.686 |
| synthesize_synopsis | 8.980 |
| make_embedding | 3.839 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 44.452 |
| branch_yolo_total | 11.433 |
| branch_audio_total | 57.488 |
