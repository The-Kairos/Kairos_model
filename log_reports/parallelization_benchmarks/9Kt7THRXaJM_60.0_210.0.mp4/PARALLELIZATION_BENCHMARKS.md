# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 17:51:17 UTC | 9Kt7THRXaJM_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 167.440 | 0.792 | 56.736 | 23.474 | 9.076 | 16.048 | 3.895 |

## 2026-06-24 17:51:17 UTC | 9Kt7THRXaJM_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/9Kt7THRXaJM_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `167.440` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.792 |
| save_clips | - |
| sample_frames | 1.224 |
| caption_frames | 43.160 |
| sample_fps | 2.330 |
| detect_object_yolo | 9.330 |
| audio_scan | 11.759 |
| asr_timings | 12.308 |
| ast_timings | 32.661 |
| describe_scenes | 23.474 |
| summarize_scenes | 9.076 |
| synthesize_synopsis | 16.048 |
| make_embedding | 3.895 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 44.390 |
| branch_yolo_total | 11.666 |
| branch_audio_total | 56.736 |
