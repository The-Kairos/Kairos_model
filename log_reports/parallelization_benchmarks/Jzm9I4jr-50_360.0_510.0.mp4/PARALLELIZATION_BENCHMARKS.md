# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 05:36:54 UTC | Jzm9I4jr-50_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 194.020 | 0.690 | 59.927 | 20.506 | 14.503 | 23.544 | 4.964 |

## 2026-06-25 05:36:54 UTC | Jzm9I4jr-50_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/Jzm9I4jr-50_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `194.020` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.690 |
| save_clips | - |
| sample_frames | 1.462 |
| caption_frames | 53.918 |
| sample_fps | 2.361 |
| detect_object_yolo | 10.716 |
| audio_scan | 7.523 |
| asr_timings | 11.579 |
| ast_timings | 40.818 |
| describe_scenes | 20.506 |
| summarize_scenes | 14.503 |
| synthesize_synopsis | 23.544 |
| make_embedding | 4.964 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 55.386 |
| branch_yolo_total | 13.083 |
| branch_audio_total | 59.927 |
