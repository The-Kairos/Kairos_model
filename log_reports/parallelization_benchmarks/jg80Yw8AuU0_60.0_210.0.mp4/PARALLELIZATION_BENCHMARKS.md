# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 11:36:50 UTC | jg80Yw8AuU0_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 188.655 | 0.680 | 59.847 | 23.477 | 11.893 | 25.924 | 4.199 |

## 2026-06-26 11:36:50 UTC | jg80Yw8AuU0_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/jg80Yw8AuU0_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `188.655` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.680 |
| save_clips | - |
| sample_frames | 1.172 |
| caption_frames | 48.060 |
| sample_fps | 2.245 |
| detect_object_yolo | 9.744 |
| audio_scan | 15.020 |
| asr_timings | 9.157 |
| ast_timings | 35.662 |
| describe_scenes | 23.477 |
| summarize_scenes | 11.893 |
| synthesize_synopsis | 25.924 |
| make_embedding | 4.199 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 49.238 |
| branch_yolo_total | 11.996 |
| branch_audio_total | 59.847 |
