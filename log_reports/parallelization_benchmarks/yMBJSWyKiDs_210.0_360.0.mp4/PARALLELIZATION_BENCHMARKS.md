# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 04:45:44 UTC | yMBJSWyKiDs_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 131.890 | 0.767 | 46.550 | 10.404 | 13.294 | 7.867 | 3.328 |

## 2026-06-27 04:45:44 UTC | yMBJSWyKiDs_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/yMBJSWyKiDs_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `131.890` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.767 |
| save_clips | - |
| sample_frames | 1.074 |
| caption_frames | 36.553 |
| sample_fps | 2.250 |
| detect_object_yolo | 8.403 |
| audio_scan | 9.768 |
| asr_timings | 9.352 |
| ast_timings | 27.421 |
| describe_scenes | 10.404 |
| summarize_scenes | 13.294 |
| synthesize_synopsis | 7.867 |
| make_embedding | 3.328 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 37.633 |
| branch_yolo_total | 10.658 |
| branch_audio_total | 46.550 |
