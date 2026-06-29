# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 20:37:14 UTC | WVfXEIyanKY_660.0_810.0.mp4 | sequential | gemini | gemini-embedding-001 | 139.470 | 0.773 | 48.298 | 6.073 | 25.264 | 11.241 | 3.033 |

## 2026-06-25 20:37:14 UTC | WVfXEIyanKY_660.0_810.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/WVfXEIyanKY_660.0_810.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `139.470` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.773 |
| save_clips | - |
| sample_frames | 0.772 |
| caption_frames | 32.752 |
| sample_fps | 2.044 |
| detect_object_yolo | 7.822 |
| audio_scan | 13.879 |
| asr_timings | 9.773 |
| ast_timings | 24.637 |
| describe_scenes | 6.073 |
| summarize_scenes | 25.264 |
| synthesize_synopsis | 11.241 |
| make_embedding | 3.033 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 33.530 |
| branch_yolo_total | 9.871 |
| branch_audio_total | 48.298 |
