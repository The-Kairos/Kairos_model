# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 13:09:14 UTC | k5pnAks5jDs_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 163.191 | 0.770 | 50.017 | 14.698 | 12.098 | 30.171 | 3.581 |

## 2026-06-26 13:09:14 UTC | k5pnAks5jDs_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/k5pnAks5jDs_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `163.191` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.770 |
| save_clips | - |
| sample_frames | 0.897 |
| caption_frames | 38.715 |
| sample_fps | 2.136 |
| detect_object_yolo | 8.653 |
| audio_scan | 10.632 |
| asr_timings | 9.766 |
| ast_timings | 29.610 |
| describe_scenes | 14.698 |
| summarize_scenes | 12.098 |
| synthesize_synopsis | 30.171 |
| make_embedding | 3.581 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 39.618 |
| branch_yolo_total | 10.795 |
| branch_audio_total | 50.017 |
