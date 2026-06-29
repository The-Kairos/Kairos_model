# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 14:21:46 UTC | 7bx2_WbwlKo_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 205.810 | 0.819 | 61.939 | 27.403 | 27.068 | 18.217 | 4.556 |

## 2026-06-24 14:21:46 UTC | 7bx2_WbwlKo_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/7bx2_WbwlKo_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `205.810` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.819 |
| save_clips | - |
| sample_frames | 1.705 |
| caption_frames | 50.030 |
| sample_fps | 2.557 |
| detect_object_yolo | 10.116 |
| audio_scan | 14.907 |
| asr_timings | 9.618 |
| ast_timings | 37.405 |
| describe_scenes | 27.403 |
| summarize_scenes | 27.068 |
| synthesize_synopsis | 18.217 |
| make_embedding | 4.556 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 51.741 |
| branch_yolo_total | 12.679 |
| branch_audio_total | 61.939 |
