# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 12:46:07 UTC | -eQfg7HJuKc_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 100.237 | 0.780 | 42.042 | 7.587 | 4.316 | 12.938 | 2.399 |

## 2026-06-27 12:46:07 UTC | -eQfg7HJuKc_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/-eQfg7HJuKc_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `100.237` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.780 |
| save_clips | - |
| sample_frames | 0.537 |
| caption_frames | 14.833 |
| sample_fps | 1.983 |
| detect_object_yolo | 6.608 |
| audio_scan | 17.720 |
| asr_timings | 9.216 |
| ast_timings | 15.097 |
| describe_scenes | 7.587 |
| summarize_scenes | 4.316 |
| synthesize_synopsis | 12.938 |
| make_embedding | 2.399 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 15.375 |
| branch_yolo_total | 8.596 |
| branch_audio_total | 42.042 |
