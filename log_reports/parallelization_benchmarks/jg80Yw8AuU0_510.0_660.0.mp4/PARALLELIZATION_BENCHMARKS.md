# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 11:33:40 UTC | jg80Yw8AuU0_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 222.189 | 0.718 | 68.442 | 24.001 | 19.738 | 27.956 | 5.397 |

## 2026-06-26 11:33:40 UTC | jg80Yw8AuU0_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/jg80Yw8AuU0_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `222.189` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.718 |
| save_clips | - |
| sample_frames | 1.747 |
| caption_frames | 59.056 |
| sample_fps | 2.593 |
| detect_object_yolo | 11.123 |
| audio_scan | 16.120 |
| asr_timings | 7.845 |
| ast_timings | 44.468 |
| describe_scenes | 24.001 |
| summarize_scenes | 19.738 |
| synthesize_synopsis | 27.956 |
| make_embedding | 5.397 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 60.809 |
| branch_yolo_total | 13.722 |
| branch_audio_total | 68.442 |
