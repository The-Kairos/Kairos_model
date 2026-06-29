# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 23:06:39 UTC | ZudB4C8rtQU_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 104.505 | 0.659 | 40.009 | 9.594 | 4.260 | 8.194 | 2.756 |

## 2026-06-25 23:06:39 UTC | ZudB4C8rtQU_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/ZudB4C8rtQU_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `104.505` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.659 |
| save_clips | - |
| sample_frames | 0.704 |
| caption_frames | 28.547 |
| sample_fps | 1.877 |
| detect_object_yolo | 6.518 |
| audio_scan | 8.410 |
| asr_timings | 10.128 |
| ast_timings | 21.462 |
| describe_scenes | 9.594 |
| summarize_scenes | 4.260 |
| synthesize_synopsis | 8.194 |
| make_embedding | 2.756 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 29.256 |
| branch_yolo_total | 8.400 |
| branch_audio_total | 40.009 |
