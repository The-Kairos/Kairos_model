# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 13:49:38 UTC | 0ikdVIvzWnY_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 94.085 | 0.793 | 38.989 | 6.045 | 5.891 | 8.675 | 2.347 |

## 2026-06-27 13:49:38 UTC | 0ikdVIvzWnY_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/0ikdVIvzWnY_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `94.085` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.793 |
| save_clips | - |
| sample_frames | 0.400 |
| caption_frames | 20.963 |
| sample_fps | 1.931 |
| detect_object_yolo | 6.644 |
| audio_scan | 13.906 |
| asr_timings | 9.517 |
| ast_timings | 15.557 |
| describe_scenes | 6.045 |
| summarize_scenes | 5.891 |
| synthesize_synopsis | 8.675 |
| make_embedding | 2.347 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 21.369 |
| branch_yolo_total | 8.580 |
| branch_audio_total | 38.989 |
