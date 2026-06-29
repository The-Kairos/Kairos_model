# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 23:04:53 UTC | ZudB4C8rtQU_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 165.959 | 0.671 | 59.000 | 13.113 | 20.352 | 7.444 | 4.135 |

## 2026-06-25 23:04:53 UTC | ZudB4C8rtQU_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/ZudB4C8rtQU_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `165.959` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.671 |
| save_clips | - |
| sample_frames | 1.248 |
| caption_frames | 46.535 |
| sample_fps | 2.223 |
| detect_object_yolo | 9.816 |
| audio_scan | 12.679 |
| asr_timings | 10.107 |
| ast_timings | 36.206 |
| describe_scenes | 13.113 |
| summarize_scenes | 20.352 |
| synthesize_synopsis | 7.444 |
| make_embedding | 4.135 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 47.789 |
| branch_yolo_total | 12.045 |
| branch_audio_total | 59.000 |
