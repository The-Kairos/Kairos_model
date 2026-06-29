# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 06:04:59 UTC | K8o5XoeNjC0_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 184.856 | 0.660 | 61.436 | 17.181 | 13.979 | 24.639 | 4.163 |

## 2026-06-25 06:04:59 UTC | K8o5XoeNjC0_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/K8o5XoeNjC0_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `184.856` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.660 |
| save_clips | - |
| sample_frames | 1.089 |
| caption_frames | 48.483 |
| sample_fps | 2.178 |
| detect_object_yolo | 9.663 |
| audio_scan | 14.871 |
| asr_timings | 10.931 |
| ast_timings | 35.627 |
| describe_scenes | 17.181 |
| summarize_scenes | 13.979 |
| synthesize_synopsis | 24.639 |
| make_embedding | 4.163 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 49.578 |
| branch_yolo_total | 11.847 |
| branch_audio_total | 61.436 |
