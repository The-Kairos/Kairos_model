# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 22:35:38 UTC | sqFIsmskWaw_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 169.942 | 0.804 | 60.462 | 13.513 | 19.390 | 7.926 | 4.127 |

## 2026-06-26 22:35:38 UTC | sqFIsmskWaw_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/sqFIsmskWaw_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `169.942` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.804 |
| save_clips | - |
| sample_frames | 1.438 |
| caption_frames | 48.561 |
| sample_fps | 2.407 |
| detect_object_yolo | 9.915 |
| audio_scan | 14.861 |
| asr_timings | 9.421 |
| ast_timings | 36.171 |
| describe_scenes | 13.513 |
| summarize_scenes | 19.390 |
| synthesize_synopsis | 7.926 |
| make_embedding | 4.127 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 50.005 |
| branch_yolo_total | 12.328 |
| branch_audio_total | 60.462 |
