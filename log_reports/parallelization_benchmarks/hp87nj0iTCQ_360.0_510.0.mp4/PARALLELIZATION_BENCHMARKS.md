# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 06:56:47 UTC | hp87nj0iTCQ_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 199.648 | 0.832 | 66.758 | 23.058 | 20.313 | 14.211 | 5.130 |

## 2026-06-26 06:56:47 UTC | hp87nj0iTCQ_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/hp87nj0iTCQ_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `199.648` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.832 |
| save_clips | - |
| sample_frames | 1.343 |
| caption_frames | 53.524 |
| sample_fps | 2.477 |
| detect_object_yolo | 10.588 |
| audio_scan | 15.029 |
| asr_timings | 10.961 |
| ast_timings | 40.760 |
| describe_scenes | 23.058 |
| summarize_scenes | 20.313 |
| synthesize_synopsis | 14.211 |
| make_embedding | 5.130 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 54.872 |
| branch_yolo_total | 13.071 |
| branch_audio_total | 66.758 |
