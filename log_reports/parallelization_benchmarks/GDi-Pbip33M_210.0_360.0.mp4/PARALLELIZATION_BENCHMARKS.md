# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 01:19:25 UTC | GDi-Pbip33M_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 125.445 | 0.812 | 40.952 | 15.448 | 11.524 | 11.890 | 2.824 |

## 2026-06-25 01:19:25 UTC | GDi-Pbip33M_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/GDi-Pbip33M_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `125.445` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.812 |
| save_clips | - |
| sample_frames | 0.933 |
| caption_frames | 29.524 |
| sample_fps | 2.228 |
| detect_object_yolo | 7.862 |
| audio_scan | 7.485 |
| asr_timings | 11.760 |
| ast_timings | 21.697 |
| describe_scenes | 15.448 |
| summarize_scenes | 11.524 |
| synthesize_synopsis | 11.890 |
| make_embedding | 2.824 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 30.463 |
| branch_yolo_total | 10.096 |
| branch_audio_total | 40.952 |
