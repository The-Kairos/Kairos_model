# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 21:51:29 UTC | CyAyEewgdEc_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 555.019 | 0.816 | 433.623 | 15.065 | 14.873 | 14.340 | 11.212 |

## 2026-06-24 21:51:29 UTC | CyAyEewgdEc_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/CyAyEewgdEc_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `555.019` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.816 |
| save_clips | - |
| sample_frames | 1.529 |
| caption_frames | 49.637 |
| sample_fps | 2.565 |
| detect_object_yolo | 9.955 |
| audio_scan | 11.810 |
| asr_timings | 383.319 |
| ast_timings | 38.486 |
| describe_scenes | 15.065 |
| summarize_scenes | 14.873 |
| synthesize_synopsis | 14.340 |
| make_embedding | 11.212 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 51.172 |
| branch_yolo_total | 12.526 |
| branch_audio_total | 433.623 |
