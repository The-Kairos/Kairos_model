# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 08:43:17 UTC | -eQfg7HJuKc_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 106.168 | 0.787 | 40.391 | 6.092 | 9.399 | 19.294 | 2.097 |

## 2026-06-24 08:43:17 UTC | -eQfg7HJuKc_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/-eQfg7HJuKc_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `106.168` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.787 |
| save_clips | - |
| sample_frames | 0.372 |
| caption_frames | 17.819 |
| sample_fps | 1.917 |
| detect_object_yolo | 6.623 |
| audio_scan | 16.058 |
| asr_timings | 11.967 |
| ast_timings | 12.357 |
| describe_scenes | 6.092 |
| summarize_scenes | 9.399 |
| synthesize_synopsis | 19.294 |
| make_embedding | 2.097 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 18.197 |
| branch_yolo_total | 8.546 |
| branch_audio_total | 40.391 |
