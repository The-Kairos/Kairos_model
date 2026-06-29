# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 05:08:15 UTC | JRkGew32t9E_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 157.434 | 0.693 | 55.762 | 18.492 | 7.040 | 16.890 | 3.601 |

## 2026-06-25 05:08:15 UTC | JRkGew32t9E_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/JRkGew32t9E_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `157.434` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.693 |
| save_clips | - |
| sample_frames | 1.007 |
| caption_frames | 41.022 |
| sample_fps | 2.155 |
| detect_object_yolo | 9.295 |
| audio_scan | 16.261 |
| asr_timings | 10.239 |
| ast_timings | 29.254 |
| describe_scenes | 18.492 |
| summarize_scenes | 7.040 |
| synthesize_synopsis | 16.890 |
| make_embedding | 3.601 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 42.035 |
| branch_yolo_total | 11.457 |
| branch_audio_total | 55.762 |
