# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 19:05:23 UTC | AV9a_rRCOaU_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 196.442 | 0.799 | 65.379 | 24.747 | 14.010 | 18.011 | 4.986 |

## 2026-06-24 19:05:23 UTC | AV9a_rRCOaU_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/AV9a_rRCOaU_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `196.442` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.799 |
| save_clips | - |
| sample_frames | 1.331 |
| caption_frames | 52.612 |
| sample_fps | 2.453 |
| detect_object_yolo | 10.687 |
| audio_scan | 9.641 |
| asr_timings | 14.410 |
| ast_timings | 41.320 |
| describe_scenes | 24.747 |
| summarize_scenes | 14.010 |
| synthesize_synopsis | 18.011 |
| make_embedding | 4.986 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 53.949 |
| branch_yolo_total | 13.146 |
| branch_audio_total | 65.379 |
