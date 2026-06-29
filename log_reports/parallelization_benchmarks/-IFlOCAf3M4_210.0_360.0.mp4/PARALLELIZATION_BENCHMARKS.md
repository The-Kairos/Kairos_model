# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 06:58:57 UTC | -IFlOCAf3M4_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 1711.684 | 0.781 | 1621.683 | 16.356 | 11.546 | 19.538 | 3.247 |

## 2026-06-24 06:58:57 UTC | -IFlOCAf3M4_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/-IFlOCAf3M4_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `1711.684` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.781 |
| save_clips | - |
| sample_frames | 1.054 |
| caption_frames | 21.512 |
| sample_fps | 2.262 |
| detect_object_yolo | 7.526 |
| audio_scan | 16.862 |
| asr_timings | 1581.503 |
| ast_timings | 23.310 |
| describe_scenes | 16.356 |
| summarize_scenes | 11.546 |
| synthesize_synopsis | 19.538 |
| make_embedding | 3.247 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 22.571 |
| branch_yolo_total | 9.793 |
| branch_audio_total | 1621.683 |
