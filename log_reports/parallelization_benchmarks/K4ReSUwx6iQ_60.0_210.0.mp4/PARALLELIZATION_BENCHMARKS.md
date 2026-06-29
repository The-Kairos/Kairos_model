# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 05:55:32 UTC | K4ReSUwx6iQ_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 184.995 | 0.781 | 98.561 | 7.685 | 8.335 | 17.090 | 3.312 |

## 2026-06-25 05:55:32 UTC | K4ReSUwx6iQ_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/K4ReSUwx6iQ_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `184.995` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.781 |
| save_clips | - |
| sample_frames | 0.998 |
| caption_frames | 36.249 |
| sample_fps | 2.202 |
| detect_object_yolo | 8.396 |
| audio_scan | 15.960 |
| asr_timings | 58.663 |
| ast_timings | 23.930 |
| describe_scenes | 7.685 |
| summarize_scenes | 8.335 |
| synthesize_synopsis | 17.090 |
| make_embedding | 3.312 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 37.252 |
| branch_yolo_total | 10.603 |
| branch_audio_total | 98.561 |
