# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 03:55:03 UTC | HsjGbDcVXn4_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 147.019 | 0.799 | 64.770 | 10.267 | 8.736 | 10.308 | 3.334 |

## 2026-06-25 03:55:03 UTC | HsjGbDcVXn4_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/HsjGbDcVXn4_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `147.019` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.799 |
| save_clips | - |
| sample_frames | 1.083 |
| caption_frames | 35.561 |
| sample_fps | 2.292 |
| detect_object_yolo | 8.463 |
| audio_scan | 13.743 |
| asr_timings | 23.933 |
| ast_timings | 27.086 |
| describe_scenes | 10.267 |
| summarize_scenes | 8.736 |
| synthesize_synopsis | 10.308 |
| make_embedding | 3.334 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 36.650 |
| branch_yolo_total | 10.761 |
| branch_audio_total | 64.770 |
