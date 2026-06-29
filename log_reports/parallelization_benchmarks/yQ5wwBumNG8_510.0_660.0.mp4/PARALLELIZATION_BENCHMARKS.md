# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 05:03:13 UTC | yQ5wwBumNG8_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 104.114 | 0.578 | 40.199 | 8.310 | 8.573 | 6.698 | 2.527 |

## 2026-06-27 05:03:13 UTC | yQ5wwBumNG8_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/yQ5wwBumNG8_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `104.114` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.578 |
| save_clips | - |
| sample_frames | 0.577 |
| caption_frames | 26.945 |
| sample_fps | 1.715 |
| detect_object_yolo | 6.503 |
| audio_scan | 12.579 |
| asr_timings | 8.866 |
| ast_timings | 18.746 |
| describe_scenes | 8.310 |
| summarize_scenes | 8.573 |
| synthesize_synopsis | 6.698 |
| make_embedding | 2.527 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 27.528 |
| branch_yolo_total | 8.224 |
| branch_audio_total | 40.199 |
