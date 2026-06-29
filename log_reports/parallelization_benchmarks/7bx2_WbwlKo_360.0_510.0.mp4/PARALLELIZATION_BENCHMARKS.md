# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 14:25:01 UTC | 7bx2_WbwlKo_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 193.532 | 0.821 | 56.672 | 25.232 | 18.022 | 27.341 | 4.221 |

## 2026-06-24 14:25:01 UTC | 7bx2_WbwlKo_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/7bx2_WbwlKo_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `193.532` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.821 |
| save_clips | - |
| sample_frames | 1.655 |
| caption_frames | 46.249 |
| sample_fps | 2.425 |
| detect_object_yolo | 9.508 |
| audio_scan | 12.796 |
| asr_timings | 8.735 |
| ast_timings | 35.132 |
| describe_scenes | 25.232 |
| summarize_scenes | 18.022 |
| synthesize_synopsis | 27.341 |
| make_embedding | 4.221 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 47.910 |
| branch_yolo_total | 11.938 |
| branch_audio_total | 56.672 |
