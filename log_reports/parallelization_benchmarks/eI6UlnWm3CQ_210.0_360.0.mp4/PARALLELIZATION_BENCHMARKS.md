# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 03:33:52 UTC | eI6UlnWm3CQ_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 173.021 | 0.598 | 67.544 | 15.653 | 13.842 | 12.081 | 4.093 |

## 2026-06-26 03:33:52 UTC | eI6UlnWm3CQ_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/eI6UlnWm3CQ_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `173.021` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.598 |
| save_clips | - |
| sample_frames | 1.401 |
| caption_frames | 44.706 |
| sample_fps | 2.032 |
| detect_object_yolo | 9.577 |
| audio_scan | 13.510 |
| asr_timings | 17.728 |
| ast_timings | 36.298 |
| describe_scenes | 15.653 |
| summarize_scenes | 13.842 |
| synthesize_synopsis | 12.081 |
| make_embedding | 4.093 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 46.112 |
| branch_yolo_total | 11.615 |
| branch_audio_total | 67.544 |
