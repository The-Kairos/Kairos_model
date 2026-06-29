# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 13:28:10 UTC | 75Tp9uriob8_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 158.084 | 0.656 | 47.364 | 19.166 | 27.978 | 13.761 | 3.326 |

## 2026-06-24 13:28:10 UTC | 75Tp9uriob8_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/75Tp9uriob8_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `158.084` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.656 |
| save_clips | - |
| sample_frames | 1.200 |
| caption_frames | 32.473 |
| sample_fps | 2.168 |
| detect_object_yolo | 8.613 |
| audio_scan | 12.734 |
| asr_timings | 7.883 |
| ast_timings | 26.740 |
| describe_scenes | 19.166 |
| summarize_scenes | 27.978 |
| synthesize_synopsis | 13.761 |
| make_embedding | 3.326 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 33.679 |
| branch_yolo_total | 10.787 |
| branch_audio_total | 47.364 |
