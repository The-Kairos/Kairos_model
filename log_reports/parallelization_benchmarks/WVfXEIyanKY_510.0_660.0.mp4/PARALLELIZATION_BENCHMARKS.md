# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 20:31:51 UTC | WVfXEIyanKY_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 142.599 | 0.799 | 54.664 | 12.259 | 8.114 | 13.558 | 3.313 |

## 2026-06-25 20:31:51 UTC | WVfXEIyanKY_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/WVfXEIyanKY_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `142.599` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.799 |
| save_clips | - |
| sample_frames | 0.928 |
| caption_frames | 36.752 |
| sample_fps | 2.146 |
| detect_object_yolo | 8.623 |
| audio_scan | 15.982 |
| asr_timings | 10.995 |
| ast_timings | 27.678 |
| describe_scenes | 12.259 |
| summarize_scenes | 8.114 |
| synthesize_synopsis | 13.558 |
| make_embedding | 3.313 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 37.686 |
| branch_yolo_total | 10.775 |
| branch_audio_total | 54.664 |
