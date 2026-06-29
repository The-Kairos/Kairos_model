# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 05:19:14 UTC | ycVc1jsaAA0_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 158.632 | 0.708 | 60.008 | 13.095 | 9.400 | 6.391 | 4.436 |

## 2026-06-27 05:19:14 UTC | ycVc1jsaAA0_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/ycVc1jsaAA0_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `158.632` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.708 |
| save_clips | - |
| sample_frames | 1.263 |
| caption_frames | 49.813 |
| sample_fps | 2.249 |
| detect_object_yolo | 9.877 |
| audio_scan | 11.517 |
| asr_timings | 10.008 |
| ast_timings | 38.475 |
| describe_scenes | 13.095 |
| summarize_scenes | 9.400 |
| synthesize_synopsis | 6.391 |
| make_embedding | 4.436 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 51.081 |
| branch_yolo_total | 12.131 |
| branch_audio_total | 60.008 |
