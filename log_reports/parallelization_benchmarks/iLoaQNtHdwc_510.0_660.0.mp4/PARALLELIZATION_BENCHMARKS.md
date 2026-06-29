# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 08:45:08 UTC | iLoaQNtHdwc_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 177.621 | 0.656 | 51.350 | 24.431 | 34.747 | 15.411 | 3.378 |

## 2026-06-26 08:45:08 UTC | iLoaQNtHdwc_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/iLoaQNtHdwc_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `177.621` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.656 |
| save_clips | - |
| sample_frames | 0.853 |
| caption_frames | 35.220 |
| sample_fps | 2.078 |
| detect_object_yolo | 8.055 |
| audio_scan | 9.794 |
| asr_timings | 13.986 |
| ast_timings | 27.560 |
| describe_scenes | 24.431 |
| summarize_scenes | 34.747 |
| synthesize_synopsis | 15.411 |
| make_embedding | 3.378 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 36.079 |
| branch_yolo_total | 10.139 |
| branch_audio_total | 51.350 |
