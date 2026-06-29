# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 11:39:27 UTC | jg80Yw8AuU0_660.0_810.0.mp4 | sequential | gemini | gemini-embedding-001 | 155.113 | 0.750 | 41.760 | 20.084 | 12.724 | 34.546 | 2.823 |

## 2026-06-26 11:39:27 UTC | jg80Yw8AuU0_660.0_810.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/jg80Yw8AuU0_660.0_810.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `155.113` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.750 |
| save_clips | - |
| sample_frames | 1.023 |
| caption_frames | 30.272 |
| sample_fps | 2.096 |
| detect_object_yolo | 7.543 |
| audio_scan | 11.999 |
| asr_timings | 8.015 |
| ast_timings | 21.737 |
| describe_scenes | 20.084 |
| summarize_scenes | 12.724 |
| synthesize_synopsis | 34.546 |
| make_embedding | 2.823 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 31.300 |
| branch_yolo_total | 9.645 |
| branch_audio_total | 41.760 |
