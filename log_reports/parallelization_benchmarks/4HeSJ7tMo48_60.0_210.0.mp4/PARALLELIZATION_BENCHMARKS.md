# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-21 22:51:25 UTC | 4HeSJ7tMo48_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 172.142 | 0.718 | 84.702 | 10.856 | 9.472 | 6.354 | 3.868 |

## 2026-06-21 22:51:25 UTC | 4HeSJ7tMo48_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/4HeSJ7tMo48_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `172.142` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.718 |
| save_clips | - |
| sample_frames | 1.097 |
| caption_frames | 42.519 |
| sample_fps | 2.153 |
| detect_object_yolo | 9.017 |
| audio_scan | 11.790 |
| asr_timings | 40.727 |
| ast_timings | 32.177 |
| describe_scenes | 10.856 |
| summarize_scenes | 9.472 |
| synthesize_synopsis | 6.354 |
| make_embedding | 3.868 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 43.622 |
| branch_yolo_total | 11.176 |
| branch_audio_total | 84.702 |
