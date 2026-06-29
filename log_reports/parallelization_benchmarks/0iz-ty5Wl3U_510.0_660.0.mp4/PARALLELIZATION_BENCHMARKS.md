# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 13:55:26 UTC | 0iz-ty5Wl3U_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 106.808 | 0.655 | 41.978 | 7.385 | 5.684 | 9.809 | 2.814 |

## 2026-06-27 13:55:26 UTC | 0iz-ty5Wl3U_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/0iz-ty5Wl3U_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `106.808` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.655 |
| save_clips | - |
| sample_frames | 0.756 |
| caption_frames | 26.977 |
| sample_fps | 1.967 |
| detect_object_yolo | 7.365 |
| audio_scan | 12.832 |
| asr_timings | 8.921 |
| ast_timings | 20.217 |
| describe_scenes | 7.385 |
| summarize_scenes | 5.684 |
| synthesize_synopsis | 9.809 |
| make_embedding | 2.814 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 27.739 |
| branch_yolo_total | 9.338 |
| branch_audio_total | 41.978 |
