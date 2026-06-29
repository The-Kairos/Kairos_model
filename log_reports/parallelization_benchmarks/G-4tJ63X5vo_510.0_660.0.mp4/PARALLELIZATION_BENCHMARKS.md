# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 00:54:30 UTC | G-4tJ63X5vo_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 135.252 | 0.597 | 55.003 | 13.899 | 7.320 | 6.417 | 3.592 |

## 2026-06-25 00:54:30 UTC | G-4tJ63X5vo_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/G-4tJ63X5vo_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `135.252` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.597 |
| save_clips | - |
| sample_frames | 0.866 |
| caption_frames | 34.959 |
| sample_fps | 2.045 |
| detect_object_yolo | 9.122 |
| audio_scan | 16.177 |
| asr_timings | 8.885 |
| ast_timings | 29.932 |
| describe_scenes | 13.899 |
| summarize_scenes | 7.320 |
| synthesize_synopsis | 6.417 |
| make_embedding | 3.592 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 35.832 |
| branch_yolo_total | 11.172 |
| branch_audio_total | 55.003 |
