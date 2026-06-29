# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 05:29:57 UTC | ysd6xzuJ6S4_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 161.691 | 0.650 | 53.406 | 13.488 | 24.219 | 10.616 | 3.896 |

## 2026-06-27 05:29:57 UTC | ysd6xzuJ6S4_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/ysd6xzuJ6S4_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `161.691` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.650 |
| save_clips | - |
| sample_frames | 1.285 |
| caption_frames | 41.134 |
| sample_fps | 2.231 |
| detect_object_yolo | 9.373 |
| audio_scan | 12.774 |
| asr_timings | 7.676 |
| ast_timings | 32.948 |
| describe_scenes | 13.488 |
| summarize_scenes | 24.219 |
| synthesize_synopsis | 10.616 |
| make_embedding | 3.896 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 42.424 |
| branch_yolo_total | 11.610 |
| branch_audio_total | 53.406 |
