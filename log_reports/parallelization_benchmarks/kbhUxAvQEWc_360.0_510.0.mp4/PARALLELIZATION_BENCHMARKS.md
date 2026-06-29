# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 14:02:27 UTC | kbhUxAvQEWc_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 229.491 | 0.669 | 54.816 | 25.576 | 45.699 | 38.931 | 4.125 |

## 2026-06-26 14:02:27 UTC | kbhUxAvQEWc_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/kbhUxAvQEWc_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `229.491` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.669 |
| save_clips | - |
| sample_frames | 1.475 |
| caption_frames | 44.754 |
| sample_fps | 2.224 |
| detect_object_yolo | 9.804 |
| audio_scan | 9.735 |
| asr_timings | 9.484 |
| ast_timings | 35.588 |
| describe_scenes | 25.576 |
| summarize_scenes | 45.699 |
| synthesize_synopsis | 38.931 |
| make_embedding | 4.125 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 46.235 |
| branch_yolo_total | 12.034 |
| branch_audio_total | 54.816 |
