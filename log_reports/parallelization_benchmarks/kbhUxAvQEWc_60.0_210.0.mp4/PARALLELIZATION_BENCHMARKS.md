# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 14:05:39 UTC | kbhUxAvQEWc_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 190.801 | 0.671 | 55.538 | 19.294 | 25.977 | 27.582 | 3.973 |

## 2026-06-26 14:05:39 UTC | kbhUxAvQEWc_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/kbhUxAvQEWc_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `190.801` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.671 |
| save_clips | - |
| sample_frames | 1.152 |
| caption_frames | 43.216 |
| sample_fps | 2.184 |
| detect_object_yolo | 9.804 |
| audio_scan | 12.851 |
| asr_timings | 9.469 |
| ast_timings | 33.210 |
| describe_scenes | 19.294 |
| summarize_scenes | 25.977 |
| synthesize_synopsis | 27.582 |
| make_embedding | 3.973 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 44.374 |
| branch_yolo_total | 11.994 |
| branch_audio_total | 55.538 |
