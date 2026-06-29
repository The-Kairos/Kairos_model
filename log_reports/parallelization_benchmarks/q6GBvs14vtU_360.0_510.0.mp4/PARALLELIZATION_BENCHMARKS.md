# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-28 08:33:50 UTC | q6GBvs14vtU_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 128.348 | 0.661 | 50.768 | 9.608 | 7.373 | 7.391 | 3.267 |

## 2026-06-28 08:33:50 UTC | q6GBvs14vtU_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/q6GBvs14vtU_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `128.348` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.661 |
| save_clips | - |
| sample_frames | 1.106 |
| caption_frames | 36.337 |
| sample_fps | 2.134 |
| detect_object_yolo | 8.313 |
| audio_scan | 13.862 |
| asr_timings | 9.821 |
| ast_timings | 27.077 |
| describe_scenes | 9.608 |
| summarize_scenes | 7.373 |
| synthesize_synopsis | 7.391 |
| make_embedding | 3.267 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 37.449 |
| branch_yolo_total | 10.452 |
| branch_audio_total | 50.768 |
