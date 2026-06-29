# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 02:03:50 UTC | GgUuz0sxwF4_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 135.009 | 0.762 | 55.156 | 6.750 | 16.583 | 10.120 | 3.003 |

## 2026-06-25 02:03:50 UTC | GgUuz0sxwF4_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/GgUuz0sxwF4_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `135.009` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.762 |
| save_clips | - |
| sample_frames | 0.715 |
| caption_frames | 30.078 |
| sample_fps | 2.093 |
| detect_object_yolo | 8.361 |
| audio_scan | 14.966 |
| asr_timings | 16.276 |
| ast_timings | 23.905 |
| describe_scenes | 6.750 |
| summarize_scenes | 16.583 |
| synthesize_synopsis | 10.120 |
| make_embedding | 3.003 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 30.799 |
| branch_yolo_total | 10.460 |
| branch_audio_total | 55.156 |
