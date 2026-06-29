# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 13:17:57 UTC | 0H2xFsZtjNY_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 117.152 | 0.834 | 49.851 | 7.818 | 8.200 | 5.909 | 2.795 |

## 2026-06-27 13:17:57 UTC | 0H2xFsZtjNY_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/0H2xFsZtjNY_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `117.152` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.834 |
| save_clips | - |
| sample_frames | 0.956 |
| caption_frames | 29.695 |
| sample_fps | 2.145 |
| detect_object_yolo | 7.526 |
| audio_scan | 12.881 |
| asr_timings | 15.624 |
| ast_timings | 21.338 |
| describe_scenes | 7.818 |
| summarize_scenes | 8.200 |
| synthesize_synopsis | 5.909 |
| make_embedding | 2.795 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 30.657 |
| branch_yolo_total | 9.678 |
| branch_audio_total | 49.851 |
