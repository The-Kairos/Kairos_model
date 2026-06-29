# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 04:43:08 UTC | IH3KQKtrJM0_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 93.911 | 0.657 | 31.338 | 4.608 | 4.653 | 33.023 | 1.516 |

## 2026-06-25 04:43:08 UTC | IH3KQKtrJM0_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/IH3KQKtrJM0_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `93.911` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.657 |
| save_clips | - |
| sample_frames | 0.152 |
| caption_frames | 9.966 |
| sample_fps | 1.499 |
| detect_object_yolo | 5.116 |
| audio_scan | 13.125 |
| asr_timings | 10.993 |
| ast_timings | 7.210 |
| describe_scenes | 4.608 |
| summarize_scenes | 4.653 |
| synthesize_synopsis | 33.023 |
| make_embedding | 1.516 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 10.124 |
| branch_yolo_total | 6.620 |
| branch_audio_total | 31.338 |
