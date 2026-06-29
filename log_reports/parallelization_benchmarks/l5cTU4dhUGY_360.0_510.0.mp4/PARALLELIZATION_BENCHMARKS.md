# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 14:43:34 UTC | l5cTU4dhUGY_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 196.555 | 0.648 | 53.780 | 16.570 | 29.402 | 31.644 | 3.848 |

## 2026-06-26 14:43:34 UTC | l5cTU4dhUGY_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/l5cTU4dhUGY_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `196.555` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.648 |
| save_clips | - |
| sample_frames | 1.177 |
| caption_frames | 46.199 |
| sample_fps | 2.222 |
| detect_object_yolo | 9.618 |
| audio_scan | 11.916 |
| asr_timings | 9.294 |
| ast_timings | 32.562 |
| describe_scenes | 16.570 |
| summarize_scenes | 29.402 |
| synthesize_synopsis | 31.644 |
| make_embedding | 3.848 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 47.382 |
| branch_yolo_total | 11.845 |
| branch_audio_total | 53.780 |
