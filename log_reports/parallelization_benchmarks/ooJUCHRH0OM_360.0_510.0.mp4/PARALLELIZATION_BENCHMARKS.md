# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-28 07:40:57 UTC | ooJUCHRH0OM_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 137.861 | 0.820 | 54.757 | 8.408 | 6.836 | 7.928 | 3.613 |

## 2026-06-28 07:40:57 UTC | ooJUCHRH0OM_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/ooJUCHRH0OM_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `137.861` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.820 |
| save_clips | - |
| sample_frames | 1.292 |
| caption_frames | 41.319 |
| sample_fps | 2.319 |
| detect_object_yolo | 9.143 |
| audio_scan | 14.913 |
| asr_timings | 9.823 |
| ast_timings | 30.012 |
| describe_scenes | 8.408 |
| summarize_scenes | 6.836 |
| synthesize_synopsis | 7.928 |
| make_embedding | 3.613 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 42.617 |
| branch_yolo_total | 11.469 |
| branch_audio_total | 54.757 |
