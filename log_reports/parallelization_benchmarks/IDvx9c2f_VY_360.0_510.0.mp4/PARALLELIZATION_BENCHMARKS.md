# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 04:21:51 UTC | IDvx9c2f_VY_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 275.107 | 0.717 | 84.440 | 20.120 | 52.454 | 12.859 | 6.628 |

## 2026-06-25 04:21:51 UTC | IDvx9c2f_VY_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/IDvx9c2f_VY_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `275.107` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.717 |
| save_clips | - |
| sample_frames | 2.095 |
| caption_frames | 76.966 |
| sample_fps | 2.821 |
| detect_object_yolo | 14.556 |
| audio_scan | 14.076 |
| asr_timings | 10.648 |
| ast_timings | 59.707 |
| describe_scenes | 20.120 |
| summarize_scenes | 52.454 |
| synthesize_synopsis | 12.859 |
| make_embedding | 6.628 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 79.067 |
| branch_yolo_total | 17.383 |
| branch_audio_total | 84.440 |
