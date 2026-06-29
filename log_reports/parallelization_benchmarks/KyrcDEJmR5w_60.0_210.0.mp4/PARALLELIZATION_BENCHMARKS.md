# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 07:02:23 UTC | KyrcDEJmR5w_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 131.532 | 0.870 | 41.521 | 13.984 | 9.126 | 28.094 | 2.330 |

## 2026-06-25 07:02:23 UTC | KyrcDEJmR5w_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/KyrcDEJmR5w_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `131.532` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.870 |
| save_clips | - |
| sample_frames | 0.794 |
| caption_frames | 24.446 |
| sample_fps | 2.071 |
| detect_object_yolo | 6.924 |
| audio_scan | 15.997 |
| asr_timings | 9.791 |
| ast_timings | 15.724 |
| describe_scenes | 13.984 |
| summarize_scenes | 9.126 |
| synthesize_synopsis | 28.094 |
| make_embedding | 2.330 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 25.246 |
| branch_yolo_total | 9.001 |
| branch_audio_total | 41.521 |
