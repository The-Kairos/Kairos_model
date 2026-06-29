# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 05:45:48 UTC | K4ReSUwx6iQ_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 158.939 | 0.758 | 83.680 | 8.908 | 20.234 | 13.083 | 2.301 |

## 2026-06-25 05:45:48 UTC | K4ReSUwx6iQ_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/K4ReSUwx6iQ_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `158.939` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.758 |
| save_clips | - |
| sample_frames | 0.546 |
| caption_frames | 18.802 |
| sample_fps | 1.971 |
| detect_object_yolo | 7.267 |
| audio_scan | 11.709 |
| asr_timings | 56.354 |
| ast_timings | 15.608 |
| describe_scenes | 8.908 |
| summarize_scenes | 20.234 |
| synthesize_synopsis | 13.083 |
| make_embedding | 2.301 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 19.355 |
| branch_yolo_total | 9.244 |
| branch_audio_total | 83.680 |
