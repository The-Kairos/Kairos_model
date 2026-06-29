# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 23:32:56 UTC | _QWKqFFxaw8_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 126.063 | 0.671 | 64.246 | 8.700 | 5.051 | 8.734 | 2.530 |

## 2026-06-25 23:32:56 UTC | _QWKqFFxaw8_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/_QWKqFFxaw8_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `126.063` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.671 |
| save_clips | - |
| sample_frames | 0.608 |
| caption_frames | 25.280 |
| sample_fps | 1.895 |
| detect_object_yolo | 6.901 |
| audio_scan | 16.445 |
| asr_timings | 28.655 |
| ast_timings | 19.137 |
| describe_scenes | 8.700 |
| summarize_scenes | 5.051 |
| synthesize_synopsis | 8.734 |
| make_embedding | 2.530 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 25.895 |
| branch_yolo_total | 8.802 |
| branch_audio_total | 64.246 |
